from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import kagglehub
from typing import Any, Text, Dict, List

# Load the dataset
path = kagglehub.dataset_download("paultimothymooney/recipenlg")
df = pd.read_csv(f"{path}/RecipeNLG_dataset.csv")

# Ensure the dataset is loaded correctly
if df.empty:
    raise ValueError("The dataset could not be loaded. Please check the file path.")

class ActionFetchRecipe(Action):
    """Fetches a recipe based on the recipe name provided by the user."""

    def name(self) -> Text:
        return "action_fetch_recipe"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the recipe name from the slot
        recipe_name = tracker.get_slot("recipe_name")

        if df.empty:
            dispatcher.utter_message(text="Sorry, I couldn't load the recipe database.")
            return []

        # Search for the recipe in the dataset
        recipe = df[df['title'].str.contains(recipe_name, case=False, na=False)]

        if recipe.empty:
            dispatcher.utter_message(text=f"Sorry, I couldn't find a recipe for {recipe_name}.")
        else:
            # Display the first matching recipe
            recipe = recipe.iloc[0]
            response = (
                f"Here's how you can make **{recipe['title']}**:\n\n"
                f"**Ingredients:**\n{recipe['ingredients']}\n\n"
                f"**Instructions:**\n{recipe['instructions']}"
            )
            dispatcher.utter_message(text=response)

        return []


class ActionListIngredients(Action):
    """Lists the ingredients for a specific recipe."""

    def name(self) -> Text:
        return "action_list_ingredients"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the recipe name from the slot
        recipe_name = tracker.get_slot("recipe_name")

        if df.empty:
            dispatcher.utter_message(text="Sorry, I couldn't load the recipe database.")
            return []

        # Search for the recipe in the dataset
        recipe = df[df['title'].str.contains(recipe_name, case=False, na=False)]

        if recipe.empty:
            dispatcher.utter_message(text=f"Sorry, I couldn't find a recipe for {recipe_name}.")
        else:
            # Display the ingredients for the first matching recipe
            recipe = recipe.iloc[0]
            response = f"Here are the ingredients for **{recipe['title']}**:\n{recipe['ingredients']}"
            dispatcher.utter_message(text=response)

        return []


class ActionProvideInstructions(Action):
    """Provides cooking instructions for a specific recipe."""

    def name(self) -> Text:
        return "action_provide_instructions"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the recipe name from the slot
        recipe_name = tracker.get_slot("recipe_name")

        if df.empty:
            dispatcher.utter_message(text="Sorry, I couldn't load the recipe database.")
            return []

        # Search for the recipe in the dataset
        recipe = df[df['title'].str.contains(recipe_name, case=False, na=False)]

        if recipe.empty:
            dispatcher.utter_message(text=f"Sorry, I couldn't find a recipe for {recipe_name}.")
        else:
            # Display the instructions for the first matching recipe
            recipe = recipe.iloc[0]
            response = f"Here are the instructions for **{recipe['title']}**:\n{recipe['instructions']}"
            dispatcher.utter_message(text=response)

        return []


class ActionSuggestRecipe(Action):
    """Suggests a random recipe to the user."""

    def name(self) -> Text:
        return "action_suggest_recipe"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if df.empty:
            dispatcher.utter_message(text="Sorry, I couldn't load the recipe database.")
            return []

        # Select a random recipe from the dataset
        recipe = df.sample(n=1).iloc[0]
        response = (
            f"How about trying this recipe: **{recipe['title']}**?\n\n"
            f"**Ingredients:**\n{recipe['ingredients']}\n\n"
            f"**Instructions:**\n{recipe['instructions']}"
        )
        dispatcher.utter_message(text=response)

        return []