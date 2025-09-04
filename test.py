import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("SPOONACULAR_API_KEY")

if not api_key:
    raise ValueError("API key is missing. Please set it in your .env file.")

# Define the base URL for Spoonacular API
base_url = "https://api.spoonacular.com/recipes/"

def search_recipes(include_ingredients, cuisine, diet, max_ready_time, num_results=5):
    """Search for recipes based on specific criteria."""
    url = f"{base_url}complexSearch"
    params = {
        "apiKey": api_key,
        "includeIngredients": include_ingredients,
        "cuisine": cuisine,
        "diet": diet,
        "maxReadyTime": max_ready_time,
        "number": num_results,
        "addRecipeInformation": True,
        "addRecipeInstructions": True,
        "addRecipeNutrition": True
    }
    response = requests.get(url, params=params)
    return response.json()

def display_recipe_info(recipe):
    """Display detailed information about a recipe."""
    print(f"Title: {recipe['title']}")
    print(f"Image URL: {recipe['image']}")
    print(f"Cooking Time: {recipe['readyInMinutes']} minutes")
    print(f"Number of Steps: {len(recipe['analyzedInstructions'][0]['steps'])}")
    
    # Nutrition information
    if 'nutrition' in recipe:
        calories = next((item['amount'] for item in recipe['nutrition']['nutrients'] if item['name'] == 'Calories'), 'N/A')
        print(f"Calories: {calories} kcal")

    # Ingredients 
    print(f"Ingredients: {display_ingredients(recipe)}")
    
    # Cooking Instructions
    if 'analyzedInstructions' in recipe:
        print("Instructions:")
        for step in recipe['analyzedInstructions'][0]['steps']:
            print(f"Step {step['number']}: {step['step']}")

def display_ingredients(recipe):
    """Display the list of ingredients used in the recipe."""
    ingredients_set = set()  

    # Accessing ingredients from analyzedInstructions
    for instruction in recipe.get('analyzedInstructions', []):
        for step in instruction.get('steps', []):
            if 'ingredients' in step and step['ingredients']:
                for ingredient in step['ingredients']:
                    name = ingredient.get('name', 'N/A')
                    ingredients_set.add(name)  
    
    return ', '.join(ingredients_set)

def main():
    """Main function to search and display recipes."""
    include_ingredients = input("Enter ingredients to include (comma-separated): ")
    cuisine = input("Enter cuisine type (e.g., Italian): ")
    diet = input("Enter diet type (e.g., vegetarian): ")
    max_ready_time = int(input("Enter maximum preparation time in minutes: "))
    
    recipes = search_recipes(include_ingredients, cuisine, diet, max_ready_time)
    
    if recipes.get('results'):
        for recipe in recipes['results']:
            print(f"\nFetching details for: {recipe['title']}")
            display_recipe_info(recipe)
    else:
        print("No recipes found.")

if __name__ == "__main__":
    main()
