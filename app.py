from fastapi import FastAPI, UploadFile, File, Request, HTTPException, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import base64
import os
import json
import re
from datetime import datetime
import asyncio
from dotenv import load_dotenv
import io
import requests

# Import OpenAI client
import openai
from openai import AsyncOpenAI

# Updated LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

if not spoonacular_api_key:
    raise ValueError("SPOONACULAR_API_KEY not found in environment variables")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain with OpenAI
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo-16k",
    openai_api_key=openai_api_key
)

# Use vision model with higher accuracy settings
vision_llm_accurate = ChatOpenAI(
    temperature=0.1, 
    model="gpt-4o",  
    openai_api_key=openai_api_key,
    max_tokens=1000  
)

# Initialize Async OpenAI client for TTS
tts_client = AsyncOpenAI(api_key=openai_api_key)

# Pydantic models
class Ingredient(BaseModel):
    name: str
    icon: str

class Recipe(BaseModel):
    id: str
    title: str
    image_url: str
    cooking_time: str
    num_steps: int
    calories: str
    ingredients: List[Ingredient]
    instructions: List[str]
    servings: int

class RecipeRequest(BaseModel):
    ingredients: List[str]
    dietary_preferences: Optional[str] = None
    cuisine_type: Optional[str] = None
    max_ready_time: Optional[int] = None

class VoiceCommand(BaseModel):
    audio_base64: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    speed: Optional[float] = 1.0

# Store active recipes and cooking sessions per user
active_recipes: Dict[str, Dict[str, Recipe]] = {}  # user_id -> recipe_id -> Recipe
cooking_sessions: Dict[str, Dict[str, dict]] = {}   # user_id -> session_id -> session

# WebSocket connections for real-time updates per user
websocket_connections: Dict[str, Dict[str, WebSocket]] = {}  # user_id -> session_id -> WebSocket

# TTS audio cache
tts_cache: Dict[str, bytes] = {}

async def generate_ingredient_icons(ingredient_names: Union[List[str], str]) -> List[Ingredient]:
    """Generate emoji icons for one or more ingredients"""
    # Convert single ingredient to list if needed
    if isinstance(ingredient_names, str):
        ingredient_names = [ingredient_names]
    
    try:
        icons = {}
        
        # Create emoji generation prompt for each ingredient
        for ingredient_name in ingredient_names:
            prompt = ChatPromptTemplate.from_template("""
            You are an AI culinary assistant. For the following food ingredient, return only a single emoji that best represents it.
            Ingredient: {ingredient}
            Return the response as a single emoji.
            """)

            # Create the chain with the prompt and LLM
            chain = prompt | llm

            # Generate the emoji for the given ingredient
            response = await chain.ainvoke({
                "ingredient": ingredient_name
            })

            # Parse response to extract emoji
            try:
                content = response.content.strip()
                
                # Validate if we received a valid emoji
                if len(content) > 3:  # emojis are typically 1-2 characters
                    raise ValueError("Received content is not a valid emoji.")
                icon = content 
                
            except ValueError as ve:
                print(f"Validation error: {ve}")
                icon = "🍽️" 

            # Store the icon for the ingredient
            icons[ingredient_name] = icon
        
        # Convert to list of Ingredient objects
        return [Ingredient(name=name, icon=icon) for name, icon in icons.items()]
        
    except Exception as e:
        print(f"Error generating ingredient icons: {e}")
        # Fallback: return ingredients with default icon
        if isinstance(ingredient_names, str):
            return [Ingredient(name=ingredient_names, icon="🍽️")]
        return [Ingredient(name=name, icon="🍽️") for name in ingredient_names]

@app.post("/api/identify-ingredients")
async def identify_ingredients(
    file: UploadFile = File(...), 
    user_id: str = Query(..., description="User ID")
):
    """Identify food ingredients from uploaded image using GPT-4 Vision with high accuracy"""
    try:
        # Read and encode image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Create enhanced vision prompt for accurate food identification
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert culinary professional with extensive knowledge of food ingredients. "
                    "Your task is to accurately identify ALL food ingredients visible in the image, "
                    "particularly those stored in freezers, refrigerators, or pantries. "
                    "Be very thorough and precise:\n"
                    "1. Identify every single food item visible, including packaged and fresh items\n"
                    "2. Look for items in containers, bags, boxes, or wrapped\n"
                    "3. Include frozen foods, vegetables, fruits, meats, dairy, condiments, etc.\n"
                    "4. Be specific about the type (e.g., 'chicken breast' not just 'chicken')\n"
                    "5. Include brand names if clearly visible\n"
                    "6. Don't include non-food items\n"
                    "Return ONLY a JSON array with objects containing ingredient names and representative emojis.\n"
                    "Format: [{\"name\": \"ingredient1\", \"icon\": \"🍗\"}, {\"name\": \"ingredient2\", \"icon\": \"🥦\"}]"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please carefully examine this image and identify ALL food ingredients visible. "
                            "Be thorough and accurate - look at every item, package, container, and food product. "
                            "Include frozen items, fresh produce, packaged goods, meats, dairy, condiments - everything edible. "
                            "For each ingredient, include an appropriate emoji icon. Return as a JSON array of objects with 'name' and 'icon' fields."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  
                        }
                    }
                ]
            }
        ]
        
        # Get response from vision model
        response = await vision_llm_accurate.ainvoke(messages)
        
        # Parse response with better error handling
        try:
            content = response.content.strip()
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                ingredients_data = json.loads(json_match.group())
            else:
                # Fallback: create basic ingredient list without icons
                ingredients_list = [item.strip() for item in content.split(",") if item.strip()]
                ingredients_data = [{"name": ingredient, "icon": "🍽️"} for ingredient in ingredients_list]
            
            # Clean and validate ingredients
            cleaned_ingredients = []
            for item in ingredients_data:
                if isinstance(item, dict):
                    name = item.get("name", "").strip().strip('"\'')
                    icon = item.get("icon", "🍽️").strip()
                else:
                    name = str(item).strip().strip('"\'')
                    icon = "🍽️"
                
                if name and len(name) > 1:
                    cleaned_ingredients.append({"name": name, "icon": icon})
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ingredients = []
            for item in cleaned_ingredients:
                if item["name"].lower() not in seen:
                    seen.add(item["name"].lower())
                    unique_ingredients.append(item)
            
        except Exception as parse_error:
            print(f"Parsing error: {parse_error}")
            # Final fallback - return a basic list
            ingredients_data = [{"name": "Unable to identify ingredients clearly. Please try another image.", "icon": "❓"}]
        
        return JSONResponse(content={"ingredients": unique_ingredients})
        
    except Exception as e:
        print(f"Error in identify_ingredients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-ingredient-icon")
async def generate_ingredient_icon(
    request: Request, 
    user_id: str = Query(..., description="User ID")
):
    """Generate an appropriate emoji icon for one or more ingredients using AI"""
    try:
        # Extract ingredients from the request body
        data = await request.json()
        ingredients = data.get("ingredients", [])
        
        # Handle both single ingredient and list of ingredients
        if isinstance(ingredients, str):
            ingredients = [ingredients]

        if not ingredients:
            raise HTTPException(status_code=400, detail="At least one ingredient is required.")
        
        # Generate icons using the improved function
        ingredients_with_icons = await generate_ingredient_icons(ingredients)
        
        # Return the icons for each ingredient
        return JSONResponse(content={"ingredients": [ingredient.model_dump() for ingredient in ingredients_with_icons]})

    except Exception as e:
        print(f"Error generating ingredient icons: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating ingredient icons: {str(e)}")


# @app.post("/api/generate-recipes")
# async def generate_recipes(
#     request: RecipeRequest, 
#     user_id: str = Query(..., description="User ID")
# ):
#     """Generate recipe suggestions based on ingredients using Spoonacular API"""
#     try:
#         # Initialize user's recipe storage if needed
#         if user_id not in active_recipes:
#             active_recipes[user_id] = {}

#         # Define the base URL for Spoonacular API
#         base_url = "https://api.spoonacular.com/recipes/complexSearch"
        
#         # Prepare parameters for the API call
#         params = {
#             "apiKey": spoonacular_api_key,
#             "includeIngredients": ",".join(request.ingredients),
#             "number": 3,
#             "addRecipeInformation": True,
#             "addRecipeInstructions": True,
#             "fillIngredients": True,
#             "addRecipeNutrition": True
#         }

#         # Dietary preference mapping
#         diet_mapping = {
#             "vegetarian": "vegetarian",
#             "vegan": "vegan",
#             "low carb": "ketogenic",  
#             "high protein": "high-protein",  
#             "no preference": None  
#         }

#         # Add optional parameters if provided
#         if request.cuisine_type:
#             params["cuisine"] = request.cuisine_type
#         if request.dietary_preferences:
#             diet_key = request.dietary_preferences.lower()
#             if diet_key in diet_mapping and diet_mapping[diet_key]:
#                 params["diet"] = diet_mapping[diet_key]
#         if request.max_ready_time:
#             params["maxReadyTime"] = request.max_ready_time
        
#         # Function to make the API request and handle missing ingredients
#         def fetch_recipes(params):
#             response = requests.get(base_url, params=params)
#             if response.status_code != 200:
#                 raise HTTPException(status_code=response.status_code, detail="Failed to fetch recipes from Spoonacular")
#             return response.json()

#         # Attempt to fetch recipes with all ingredients
#         data = fetch_recipes(params)
        
#         # If no recipes found, try removing one ingredient at a time
#         if not data.get('results'):
#             for i in range(len(request.ingredients)):
#                 params["includeIngredients"] = ",".join(request.ingredients[:i] + request.ingredients[i+1:])
#                 data = fetch_recipes(params)
#                 if data.get('results'):
#                     break
        
#         # If still no recipes found, raise an exception
#         if not data.get('results'):
#             raise HTTPException(status_code=404, detail="No recipes found with the given ingredients")
        
#         # Format the response to match the expected structure
#         recipes = []
#         for i, recipe_data in enumerate(data.get('results', [])):
#             # Extract calories from nutrition data
#             calories = "N/A"
#             if 'nutrition' in recipe_data:
#                 for nutrient in recipe_data['nutrition']['nutrients']:
#                     if nutrient['name'] == 'Calories':
#                         calories = f"{nutrient['amount']} {nutrient['unit']}"
#                         break
            
#             # Extract ingredients
#             ingredient_names = []
#             for ing in recipe_data.get('extendedIngredients', []):
#                 ingredient_names.append(f"{ing.get('name', '')}")
            
#             # Generate icons for ingredients
#             ingredients_with_icons = await generate_ingredient_icons(ingredient_names)
            
#             # Extract instructions
#             instructions = []
#             if recipe_data.get('analyzedInstructions') and len(recipe_data['analyzedInstructions']) > 0:
#                 for step in recipe_data['analyzedInstructions'][0].get('steps', []):
#                     instructions.append(f"Step {step.get('number', '')}: {step.get('step', '')}")
            
#             # Create recipe object
#             recipe_id = f"recipe_{user_id}_{i+1}"
#             recipe = Recipe(
#                 id=recipe_id,
#                 title=recipe_data.get('title', 'Unknown Recipe'),
#                 image_url=recipe_data.get('image', ''),
#                 cooking_time=f"{recipe_data.get('readyInMinutes', 0)} minutes",
#                 num_steps=len(instructions),
#                 calories=calories,
#                 ingredients=ingredients_with_icons,
#                 instructions=instructions,
#                 servings=recipe_data.get('servings', 1)
#             )
            
#             recipes.append(recipe)
#             active_recipes[user_id][recipe_id] = recipe
        
#         return JSONResponse(content={"recipes": [recipe.model_dump() for recipe in recipes]})
        
#     except Exception as e:
#         print(f"Error in generate_recipes: {e}")
#         raise HTTPException(status_code=500, detail=f"Error generating recipes: {str(e)}")

@app.post("/api/generate-recipes")
async def generate_recipes(
    request: RecipeRequest, 
    user_id: str = Query(..., description="User ID")
):
    """Generate recipe suggestions based on ingredients using Spoonacular API's findByIngredients endpoint"""
    try:
        if user_id not in active_recipes:
            active_recipes[user_id] = {}

        # Use findByIngredients endpoint
        base_url = "https://api.spoonacular.com/recipes/findByIngredients"
        
        params = {
            "apiKey": spoonacular_api_key,
            "ingredients": ",".join(request.ingredients),  # Comma-separated list
            "number": 100,  # Get more recipes initially to filter later
            "ranking": 2,  # Maximize used ingredients
            "ignorePantry": True,  # Exclude pantry staples
        }

        # Make initial API request
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch recipes from Spoonacular")
        
        recipe_list = response.json()
        
        # If no recipes found, return empty list
        if not recipe_list:
            return JSONResponse(content={"recipes": []})
        
        # Extract recipe IDs for detailed information
        recipe_ids = [str(recipe['id']) for recipe in recipe_list]
        
        # Get detailed recipe information using bulk endpoint
        detailed_url = "https://api.spoonacular.com/recipes/informationBulk"
        detailed_params = {
            "apiKey": spoonacular_api_key,
            "ids": ",".join(recipe_ids),
            "includeNutrition": True
        }
        
        detailed_response = requests.get(detailed_url, params=detailed_params)
        if detailed_response.status_code != 200:
            raise HTTPException(status_code=detailed_response.status_code, detail="Failed to fetch recipe details")
        
        detailed_recipes = detailed_response.json()
        
        # Apply filters
        filtered_recipes = []
        for recipe_data in detailed_recipes:
            # Filter by cuisine
            if request.cuisine_type:
                cuisines = recipe_data.get('cuisines', [])
                if not any(request.cuisine_type.lower() in cuisine.lower() for cuisine in cuisines):
                    continue
            
            # Filter by dietary preferences
            if request.dietary_preferences and request.dietary_preferences.lower() != "no preference":
                diet_key = request.dietary_preferences.lower()
                
                if diet_key == "vegetarian" and not recipe_data.get('vegetarian', False):
                    continue
                if diet_key == "vegan" and not recipe_data.get('vegan', False):
                    continue
                if diet_key == "low carb":
                    if 'nutrition' in recipe_data:
                        carbs = next((n for n in recipe_data['nutrition']['nutrients'] if n['name'] == 'Carbohydrates'), None)
                        if carbs and carbs['amount'] > 100:  # More generous threshold
                            continue

                if diet_key == "high protein":
                    if 'nutrition' in recipe_data:
                        protein = next((n for n in recipe_data['nutrition']['nutrients'] if n['name'] == 'Protein'), None)
                        if protein and protein['amount'] < 15:  # Lower threshold
                            continue
            
            # Filter by max ready time
            if request.max_ready_time and recipe_data.get('readyInMinutes', 0) > request.max_ready_time:
                continue
            
            filtered_recipes.append(recipe_data)
        
        # Limit to 3 recipes
        filtered_recipes = filtered_recipes[:3]
        
        # Format response
        recipes = []
        for i, recipe_data in enumerate(filtered_recipes):
            # Extract calories
            calories = "N/A"
            if 'nutrition' in recipe_data:
                for nutrient in recipe_data['nutrition']['nutrients']:
                    if nutrient['name'] == 'Calories':
                        calories = f"{nutrient['amount']} {nutrient['unit']}"
                        break
            
            # Extract ingredients
            ingredient_names = [ing.get('name', '') for ing in recipe_data.get('extendedIngredients', [])]
            ingredients_with_icons = await generate_ingredient_icons(ingredient_names)
            
            # Extract instructions
            instructions = []
            if recipe_data.get('analyzedInstructions') and len(recipe_data['analyzedInstructions']) > 0:
                for step in recipe_data['analyzedInstructions'][0].get('steps', []):
                    instructions.append(f"Step {step.get('number', '')}: {step.get('step', '')}")
            
            # Create recipe object
            recipe_id = f"recipe_{user_id}_{i+1}"
            recipe = Recipe(
                id=recipe_id,
                title=recipe_data.get('title', 'Unknown Recipe'),
                image_url=recipe_data.get('image', ''),
                cooking_time=f"{recipe_data.get('readyInMinutes', 0)} minutes",
                num_steps=len(instructions),
                calories=calories,
                ingredients=ingredients_with_icons,
                instructions=instructions,
                servings=recipe_data.get('servings', 1)
            )
            
            recipes.append(recipe)
            active_recipes[user_id][recipe_id] = recipe
        
        return JSONResponse(content={"recipes": [recipe.model_dump() for recipe in recipes]})
        
    except Exception as e:
        print(f"Error in generate_recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recipes: {str(e)}")


@app.post("/api/start-cooking/{recipe_id}")
async def start_cooking(
    recipe_id: str, 
    user_id: str = Query(..., description="User ID")
):
    """Start a cooking session for a specific recipe"""
    # Initialize user-specific storage if needed
    if user_id not in active_recipes:
        active_recipes[user_id] = {}
    
    if recipe_id not in active_recipes[user_id]:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe = active_recipes[user_id][recipe_id]
    session_id = f"session_{datetime.now().timestamp()}"
    
    # Initialize user-specific cooking sessions if needed
    if user_id not in cooking_sessions:
        cooking_sessions[user_id] = {}
    
    # Store session without pre-generated TTS to improve response time
    cooking_sessions[user_id][session_id] = {
        "recipe_id": recipe_id,
        "current_step": 0,
        "timers": [],
        "started_at": datetime.now().isoformat()
    }
    
    # Initialize user-specific WebSocket connections if needed
    if user_id not in websocket_connections:
        websocket_connections[user_id] = {}
    
    return {
        "session_id": session_id,
        "recipe": recipe.model_dump(),
        "current_step": 0,
        "total_steps": len(recipe.instructions)
    }

@app.get("/api/tts/{audio_key}")
async def get_tts_audio(audio_key: str):
    """Get TTS audio by key"""
    if audio_key not in tts_cache:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_content = tts_cache[audio_key]
    
    # Create an in-memory file-like object
    audio_file = io.BytesIO(audio_content)
    
    return StreamingResponse(
        audio_file,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename={audio_key}.mp3"
        }
    )

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text"""
    try:
        # Create cache key from text content
        cache_key = f"tts_{hash(request.text)}"
        
        # Check cache first
        if cache_key in tts_cache:
            audio_content = tts_cache[cache_key]
        else:
            # Generate new TTS
            response = await tts_client.audio.speech.create(
                model="tts-1",
                voice=request.voice,
                input=request.text,
                speed=request.speed
            )
            audio_content = await response.aread()
            tts_cache[cache_key] = audio_content
        
        # Create an in-memory file-like object
        audio_file = io.BytesIO(audio_content)
        
        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_audio.mp3"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """WebSocket for real-time cooking guidance with continuous voice detection"""
    await websocket.accept()
    
    # Initialize user-specific WebSocket connections if needed
    if user_id not in websocket_connections:
        websocket_connections[user_id] = {}
    
    websocket_connections[user_id][session_id] = websocket
    
    try:
        # Get session and recipe
        if user_id not in cooking_sessions or session_id not in cooking_sessions[user_id]:
            await websocket.send_json({"error": "Session not found"})
            return
            
        session = cooking_sessions[user_id][session_id]
        
        if user_id not in active_recipes or session["recipe_id"] not in active_recipes[user_id]:
            await websocket.send_json({"error": "Recipe not found"})
            return
            
        recipe = active_recipes[user_id][session["recipe_id"]]
        current_step = session["current_step"]
        
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "command")
            
            # Handle continuous voice data
            if message_type == "voice_data":
                audio_base64 = data.get("audio_base64")
                if audio_base64:
                    try:
                        # Decode and transcribe
                        audio_bytes = base64.b64decode(audio_base64)
                        temp_file = f"temp_audio_{session_id}_{datetime.now().timestamp()}.webm"
                        
                        with open(temp_file, "wb") as f:
                            f.write(audio_bytes)
                        
                        # Transcribe using Whisper
                        client = AsyncOpenAI(api_key=openai_api_key)
                        with open(temp_file, "rb") as audio_file:
                            transcript = await client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
                        # Parse command from transcription
                        text = transcript.text.lower()
                        command = None
                        
                        if any(word in text for word in ['next', 'continue', 'forward']):
                            command = 'next'
                        elif any(word in text for word in ['previous', 'back', 'go back']):
                            command = 'previous'
                        elif 'repeat' in text:
                            command = 'repeat'
                        elif 'timer' in text:
                            # Extract timer duration
                            match = re.search(r'(\d+)\s*(minute|min|second|sec)', text)
                            if match:
                                duration = match.group(1)
                                unit = match.group(2)
                                command = f'timer {duration} {unit}'
                        
                        if command:
                            await websocket.send_json({
                                "action": "voice_recognized",
                                "transcription": transcript.text,
                                "command": command
                            })
                            # Process the command
                            data = {"command": command}
                        else:
                            await websocket.send_json({
                                "action": "voice_not_recognized",
                                "transcription": transcript.text
                            })
                            continue
                            
                    except Exception as e:
                        print(f"Voice processing error: {e}")
                        await websocket.send_json({
                            "action": "voice_error",
                            "error": str(e)
                        })
                        continue
            
            # Process commands
            command = data.get("command", "").lower()
            
            # Validate command sequence
            if command == "next" and current_step >= len(recipe.instructions) - 1:
                await websocket.send_json({
                    "action": "recipe_completed",
                    "message": "Congratulations! You've completed the recipe!"
                })
                continue
                
            if command == "previous" and current_step <= 0:
                await websocket.send_json({
                    "error": "Already at first step"
                })
                continue
                
            # Process valid commands
            if command == "next":
                current_step += 1
                session["current_step"] = current_step
                
                await websocket.send_json({
                    "action": "step_updated",
                    "current_step": current_step,
                    "step_text": recipe.instructions[current_step],
                    "is_last_step": current_step == len(recipe.instructions) - 1
                })
                
            elif command == "previous" or command == "back":
                current_step -= 1
                session["current_step"] = current_step
                
                await websocket.send_json({
                    "action": "step_updated",
                    "current_step": current_step,
                    "step_text": recipe.instructions[current_step],
                    "is_last_step": False
                })
                
            elif command == "repeat":
                await websocket.send_json({
                    "action": "step_repeated",
                    "current_step": current_step,
                    "step_text": recipe.instructions[current_step]
                })
                
            elif command.startswith("timer"):
                # Extract timer duration from command
                try:
                    match = re.search(r'(\d+)\s*(minute|min|second|sec)', command)
                    if match:
                        duration = int(match.group(1))
                        unit = match.group(2)
                        seconds = duration * 60 if 'min' in unit else duration
                        
                        timer_id = f"timer_{datetime.now().timestamp()}"
                        session["timers"].append({
                            "id": timer_id,
                            "duration": seconds,
                            "started_at": datetime.now().isoformat()
                        })
                        
                        await websocket.send_json({
                            "action": "timer_started",
                            "timer_id": timer_id,
                            "duration": seconds
                        })
                        
                        # Start timer countdown
                        asyncio.create_task(handle_timer(websocket, timer_id, seconds))
                except:
                    await websocket.send_json({
                        "error": "Could not parse timer duration"
                    })
                    
            elif command == "status":
                await websocket.send_json({
                    "action": "status",
                    "current_step": current_step,
                    "total_steps": len(recipe.instructions),
                    "step_text": recipe.instructions[current_step],
                    "timers": session["timers"]
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if user_id in websocket_connections and session_id in websocket_connections[user_id]:
            del websocket_connections[user_id][session_id]

async def handle_timer(websocket: WebSocket, timer_id: str, duration: int):
    """Handle timer countdown"""
    await asyncio.sleep(duration)
    try:
        await websocket.send_json({
            "action": "timer_completed",
            "timer_id": timer_id,
            "message": "Timer completed!"
        })
    except:
        pass

@app.post("/api/voice-command/{session_id}")
async def process_voice_command(
    session_id: str, 
    voice_data: VoiceCommand,
    user_id: str = Query(..., description="User ID")
):
    """Process voice commands using Whisper API"""
    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(voice_data.audio_base64)
        
        # Save temporarily
        temp_file = f"temp_audio_{session_id}.webm"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Transcribe using Whisper
        with open(temp_file, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {
            "transcribed_text": transcript.text,
            "status": "processed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recipe/{recipe_id}")
async def get_recipe(
    recipe_id: str, 
    user_id: str = Query(..., description="User ID")
):
    """Get a specific recipe by ID"""
    if user_id not in active_recipes or recipe_id not in active_recipes[user_id]:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    return active_recipes[user_id][recipe_id].model_dump()

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)