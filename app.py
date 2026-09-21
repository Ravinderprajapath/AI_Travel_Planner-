import os
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
import os
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

class Activity(BaseModel):
    name: str = Field(description="Name of the activity")
    description: str = Field(description="Short description of the activity")
    estimated_cost: str = Field(description="Estimated cost")

class DayPlan(BaseModel):
    day: int = Field(description="Day number")
    activities: List[Activity] = Field(description="Activities for the day")
    food_suggestion: List[str] = Field(description="Food suggestions for the day")
    hotel_suggestion: str = Field(description="Hotel/accommodation suggestion")

class BudgetBreakdown(BaseModel):
    accommodation: str
    food: str
    transport: str
    activities: str
    miscellaneous: str
    total_estimated: str

class TravelItinerary(BaseModel):
    destination: str
    overview: str
    duration: str
    budget: str
    interests: List[str]
    itinerary: List[DayPlan]
    estimated_budget: BudgetBreakdown
    travel_tips: List[str]

st.title(":rainbow[AI Travel Planner]")
st.subheader("Plan your personalized trip using Google Gemini + LangChain")
st.markdown(
    "Create a personalized day-by-day travel itinerary based on your "
    "destination, budget, duration, and interests."
)

with st.sidebar:
    st.header(" Trip Settings")

    destination = st.text_input(
        " Destination",
        value="Hyderabad, India"
    )

    budget = st.text_input(
        " Budget",
        value="15,000"
    )

    duration = st.number_input(
        " Duration (Days)",
        min_value=1,
        max_value=30,
        value=3,
        step=1
    )

    interests = st.multiselect(
        " Interests",
        options=[
            "Food",
            "Culture and History",
            "Shopping",
            "Nature",
            "Adventure",
            "Beaches",
            "Nightlife",
            "Photography",
            "Museums",
            "Spirituality",
            "Luxury",
            "Local Experiences"
        ],
        default=[
            "Food",
            "Culture and History",
            "Shopping"
        ]
    )

    generate_button = st.button(
        " Generate Itinerary",
        use_container_width=True
    )

prompt_template = """
You are an expert AI travel planner.

Create a personalized travel itinerary.

Destination: {destination}
Budget: {budget}
Duration: {duration} days
Interests: {interests}

Instructions:
1. Provide a destination overview.
2. Create a day-wise itinerary for exactly {duration} days.
3. Suggest activities based on the user's interests.
4. Suggest local food and restaurants.
5. Provide an estimated budget breakdown.
6. Include accommodation, food, transport, activities, and miscellaneous expenses.
7. Provide useful travel tips.
8. Make the itinerary practical and personalized.
9. Use the user's budget and currency.
10. Mention that prices are estimates and may change.
11. Do not invent exact live prices or reservations.
12. Make sure the itinerary contains exactly {duration} days.

Return ONLY the structured output.

{format_instructions}
"""

if generate_button:
    try:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
    except Exception:
        st.error(
            " Google Gemini API key not found. "
            "Create .streamlit/secrets.toml and add "
            'GOOGLE_API_KEY = "your-api-key-here".'
        )
        st.stop()

    if not destination:
        st.error(" Please enter a destination.")
    elif not budget:
        st.error(" Please enter your budget.")
    elif not interests:
        st.error(" Please select at least one interest.")
    else:
        try:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

            parser = PydanticOutputParser(
                pydantic_object=TravelItinerary
            )

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[
                    "destination",
                    "budget",
                    "duration",
                    "interests"
                ],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                }
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                temperature=0.7,
                max_retries=5
            )

            travel_chain = prompt | llm | parser

            with st.spinner(" Generating your travel itinerary..."):
                result = travel_chain.invoke({
                    "destination": destination,
                    "budget": budget,
                    "duration": duration,
                    "interests": ", ".join(interests)
                })

            st.success(" Travel itinerary generated successfully!")

            st.header(f" {result.destination}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Duration", result.duration)

            with col2:
                st.metric("Budget", result.budget)

            with col3:
                st.metric("Interests", len(result.interests))

            st.subheader(" Trip Overview")
            st.write(result.overview)

            st.subheader(" Your Interests")
            for interest in result.interests:
                st.markdown(f"• {interest}")

            st.header(" Day-by-Day Itinerary")

            for day in result.itinerary:
                with st.expander(f" Day {day.day}", expanded=True):
                    st.subheader(" Activities")

                    for activity in day.activities:
                        st.markdown(f"###  {activity.name}")
                        st.write(activity.description)
                        st.info(
                            f" Estimated Cost: {activity.estimated_cost}"
                        )

                    st.subheader(" Food Suggestions")
                    for food in day.food_suggestion:
                        st.markdown(f"• {food}")

                    st.subheader(" Accommodation")
                    st.write(day.hotel_suggestion)

            st.header(" Estimated Budget")

            budget_data = result.estimated_budget

            col1, col2 = st.columns(2)

            with col1:
                st.metric(" Accommodation", budget_data.accommodation)
                st.metric(" Food", budget_data.food)
                st.metric(" Transport", budget_data.transport)

            with col2:
                st.metric(" Activities", budget_data.activities)
                st.metric(" Miscellaneous", budget_data.miscellaneous)
                st.metric(" Total Estimated", budget_data.total_estimated)

            st.header(" Travel Tips")

            for tip in result.travel_tips:
                st.markdown(f"• {tip}")

            with st.expander(" View Structured Response"):
                st.json(result.model_dump())

        except Exception as e:
            st.error(
                " Something went wrong while generating the itinerary."
            )
            st.exception(e)

else:
    st.info(
        " Enter your trip details in the sidebar and click "
        "**Generate Itinerary**."
    )

    st.markdown(
        """
        ###  Features

        -  Google Gemini powered
        -  LangChain integration
        -  Structured Pydantic output
        -  Day-wise itinerary
        -  Food recommendations
        -  Accommodation suggestions
        -  Budget breakdown
        -  Travel tips
        """
    )
