import os

gem = input("Enter your Gemini API key: ")
os.environ["GOOGLE_API_KEY"] = gem


from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

#create pydantic output model

class Activity(BaseModel):
  name: str = Field(
      description = "Name of the activity"
  )

  description: str = Field(
      description = " short Description"
  )

  estimated_cost: str = Field(
      description="Estimated Cost"
  )

class DayPlan(BaseModel):
  day: int = Field(
      description="Day number"
  )

  activities: List[Activity] = Field(
      description="Activities for the day"
  )

  food_suggestion: List[str] = Field(
      description="Food suggestion for the day"
  )

  hotel_suggestion: str = Field(
      description="Food suggestion"
  )

class BudgetBreakdown(BaseModel):
  accommodation: str
  food: str
  transport: str
  activities: str
  miscellaneous: str
  total_estimated: str

class TravelItinerary(BaseModel):
  destination:str

  overview: str

  duration: str

  budget: str

  interests: List[str]

  itinerary: List[DayPlan]

  estimated_budget:  BudgetBreakdown

  travel_tips: List[str]

parser = PydanticOutputParser(pydantic_object=TravelItinerary)
print("pydantic output parser created")

# create prompt template
prompt_template ="""
you are an expert AI travel planner

create a personalized travel itinerary.

Destination: {destination}

Budget: {budget}

Duration: {duration} days

Interests: {interests}

Instruction:

1. provide a destion overview.

2. create a day-wise itinerary for exactly {duration} days.

3. suggest activities based on the interests.

4. suggest local food and restaurants.

5. provide an estimated budget breakdown.

6. Include accommadation, food, transport,
   activities, and miscellaneous expenses.

7. provide usefull travel tips.

8. make the itinerary practical and personalized.

9. use the user's budget and currency.

10. Mention that prices are estimates and may change.

11. do not invent exact live prices or reservations.

return ONLY the structured output.

{format_instructions}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["destination",
                     "budget",
                     "duration",
                     "interests"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    max_retries=5,
)
print('google gemini model initialized!')

# chain
travel_chain = prompt | llm | parser
print("LangChain chain created successfully!")

# Genrate a travel itinerary

destination = "Hyderabad, India"

budget = "15,000"

duration = 3

interests = [
    "Food",
    "Culture and History",
    "Shopping"
]

result = travel_chain.invoke({
    "destination": destination,
    "budget": budget,
    "duration": duration,
    "interests": ", ".join(interests),
})

print("Travel itinerary generated successfully")

def print_itinerary(itinerary):
    print("\n" + "=" * 60)
    print(f"{itinerary.destination} Itinerary ({itinerary.duration})")
    print("=" * 60)

    print(f"Budget: {itinerary.budget}")
    print(f"Interests: {', '.join(itinerary.interests)}")

    print("\nOverview:")
    print(itinerary.overview)

    for day in itinerary.itinerary:
        print(f"\n{'-' * 50}")
        print(f"DAY {day.day}")
        print(f"{'-' * 50}")

        for act in day.activities:
            print(f"\n📌 {act.name}")
            print(f"Description: {act.description}")
            print(f"Estimated Cost: {act.estimated_cost}")

        print("\nFood Suggestions:")
        for food in day.food_suggestion:
            print(f"  - {food}")

        print(f"\n Hotel: {day.hotel_suggestion}")

    print("\n" + "=" * 60)
    print("ESTIMATED BUDGET")
    print("=" * 60)

    budget = itinerary.estimated_budget

    print(f"Accommodation: {budget.accommodation}")
    print(f"Food:          {budget.food}")
    print(f"Transport:     {budget.transport}")
    print(f"Activities:    {budget.activities}")
    print(f"Miscellaneous: {budget.miscellaneous}")
    print(f"Total:         {budget.total_estimated}")

    print("\n" + "=" * 60)
    print(" TRAVEL TIPS")
    print("=" * 60)

    for tip in itinerary.travel_tips:
        print(f"- {tip}")

print_itinerary(result)
