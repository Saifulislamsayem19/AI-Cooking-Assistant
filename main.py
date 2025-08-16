from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import os
import json
import re
from datetime import datetime
import asyncio
from dotenv import load_dotenv
import io

# Import OpenAI client
import openai
from openai import AsyncOpenAI

# Updated LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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
    model="gpt-4o-mini",
    openai_api_key=openai_api_key
)

vision_llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=openai_api_key
)

# Initialize Async OpenAI client for TTS
tts_client = AsyncOpenAI(api_key=openai_api_key)

# Pydantic models
class IngredientList(BaseModel):
    ingredients: List[Dict[str, str]]

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    cooking_time: str
    servings: int

class RecipeRequest(BaseModel):
    ingredients: List[str]
    dietary_preferences: Optional[str] = None
    cuisine_type: Optional[str] = None

class VoiceCommand(BaseModel):
    audio_base64: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    speed: Optional[float] = 1.0

# Store active recipes and cooking sessions
active_recipes: Dict[str, Recipe] = {}
cooking_sessions: Dict[str, dict] = {}

# WebSocket connections for real-time updates
websocket_connections: Dict[str, WebSocket] = {}

# TTS audio cache
tts_cache: Dict[str, bytes] = {}

@app.post("/api/identify-ingredients")
async def identify_ingredients(file: UploadFile = File(...)):
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
                    "Return ONLY a JSON array with ingredient names as strings.\n"
                    "Format: [\"ingredient1\", \"ingredient2\", \"ingredient3\"]"
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
                            "Return as a JSON array of ingredient names."
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
        
        # Use vision model with higher accuracy settings
        vision_llm_accurate = ChatOpenAI(
            temperature=0.1, 
            model="gpt-4o",  
            openai_api_key=openai_api_key,
            max_tokens=1000  
        )
        
        # Get response from vision model
        response = await vision_llm_accurate.ainvoke(messages)
        
        # Parse response with better error handling
        try:
            content = response.content.strip()
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                ingredients_list = json.loads(json_match.group())
            else:
                # Fallback: parse as comma-separated list
                ingredients_list = [item.strip() for item in content.split(",") if item.strip()]
            
            # Clean and validate ingredients
            cleaned_ingredients = []
            for ingredient in ingredients_list:
                if isinstance(ingredient, str) and ingredient.strip():
                    # Remove any quotes or special characters
                    clean_name = ingredient.strip().strip('"\'')
                    if clean_name and len(clean_name) > 1:  
                        cleaned_ingredients.append(clean_name)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ingredients = []
            for item in cleaned_ingredients:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique_ingredients.append(item)
            
            # Format response without quantity_grams
            ingredients_data = [{"name": ingredient} for ingredient in unique_ingredients]
            
        except Exception as parse_error:
            print(f"Parsing error: {parse_error}")
            # Final fallback - return a basic list
            ingredients_data = [{"name": "Unable to identify ingredients clearly. Please try another image."}]
        
        return JSONResponse(content={"ingredients": ingredients_data})
        
    except Exception as e:
        print(f"Error in identify_ingredients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipes")
async def generate_recipes(request: RecipeRequest):
    """Generate 3 recipe suggestions based on ingredients using modern LangChain LCEL syntax"""
    try:
        # Create recipe generation prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a professional chef. Based on the following ingredients, generate exactly 3 different recipe suggestions.
        
        Ingredients: {ingredients}
        Dietary Preferences: {dietary_preferences}
        Cuisine Type: {cuisine_type}
        
        Return the response as a JSON object with a 'recipes' array containing 3 recipes.
        Each recipe should have:
        - id: unique identifier (use recipe1, recipe2, recipe3)
        - title: recipe name
        - ingredients: list of ingredients with quantities
        - steps: detailed step-by-step cooking instructions
        - cooking_time: total time needed
        - servings: number of servings
        
        Make sure the recipes are diverse and creative. Ensure the JSON is properly formatted.
        """)
        
        # Create the chain with the prompt and LLM
        chain = prompt | llm
        
        response = await chain.ainvoke({
            "ingredients": ", ".join(request.ingredients),
            "dietary_preferences": request.dietary_preferences or "None",
            "cuisine_type": request.cuisine_type or "Any"
        })
        
        # Parse response
        try:
            content = response.content.strip()
            
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                recipes_data = json.loads(json_match.group())
            else:
                # If no JSON found, try parsing the entire content
                recipes_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response.content}")
        
        # Store recipes in memory
        for recipe_data in recipes_data.get('recipes', []):
            recipe = Recipe(**recipe_data)
            active_recipes[recipe.id] = recipe
        
        return JSONResponse(content=recipes_data)
        
    except Exception as e:
        print(f"Error in generate_recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recipes: {str(e)}")

@app.post("/api/start-cooking/{recipe_id}")
async def start_cooking(recipe_id: str):
    """Start a cooking session for a specific recipe"""
    if recipe_id not in active_recipes:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe = active_recipes[recipe_id]
    session_id = f"session_{datetime.now().timestamp()}"
    
    # Store session without pre-generated TTS to improve response time
    cooking_sessions[session_id] = {
        "recipe_id": recipe_id,
        "current_step": 0,
        "timers": [],
        "started_at": datetime.now().isoformat()
    }
    
    return {
        "session_id": session_id,
        "recipe": recipe.model_dump(),  
        "current_step": 0,
        "total_steps": len(recipe.steps)
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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time cooking guidance with continuous voice detection"""
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    try:
        # Get session and recipe
        if session_id not in cooking_sessions:
            await websocket.send_json({"error": "Session not found"})
            return
            
        session = cooking_sessions[session_id]
        recipe = active_recipes[session["recipe_id"]]
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
            if command == "next" and current_step >= len(recipe.steps) - 1:
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
                    "step_text": recipe.steps[current_step],
                    "is_last_step": current_step == len(recipe.steps) - 1
                })
                
            elif command == "previous" or command == "back":
                current_step -= 1
                session["current_step"] = current_step
                
                await websocket.send_json({
                    "action": "step_updated",
                    "current_step": current_step,
                    "step_text": recipe.steps[current_step],
                    "is_last_step": False
                })
                
            elif command == "repeat":
                await websocket.send_json({
                    "action": "step_repeated",
                    "current_step": current_step,
                    "step_text": recipe.steps[current_step]
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
                    "total_steps": len(recipe.steps),
                    "step_text": recipe.steps[current_step],
                    "timers": session["timers"]
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if session_id in websocket_connections:
            del websocket_connections[session_id]

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
async def process_voice_command(session_id: str, voice_data: VoiceCommand):
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
async def get_recipe(recipe_id: str):
    """Get a specific recipe by ID"""
    if recipe_id not in active_recipes:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    return active_recipes[recipe_id].model_dump()  

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)