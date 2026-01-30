from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Schema
# class Product_review(TypedDict):
#     summary : str
#     sentiment : str

# Annoted schema
class Product_review(TypedDict):
    key_themes : Annotated[list[str],"Write down all the key themes discussed in review in a list"]
    summary : Annotated[str,"A brief summary of the review"]
    # sentiment : Annotated[str,"Return sentiment of the review either neagtive, positive or neutral"]
    sentiment : Annotated[Literal["positive","negative","neutral"],"Return sentiment of the review either neagtive, positive or neutral"]
    pros:Annotated[Optional[list[str]], "Write down all the pros"]
    cons:Annotated[Optional[list[str]], "Write down all the cons"]

st_model = model.with_structured_output(Product_review)
review = """I purchased a brand new iPhone 17 Pro Max from Amazon, and within days of delivery the device started showing a visible red/dead pixel issue with intermittent flickering. The defect is clearly visible to the naked eye and appears across multiple screens and colors.

I immediately followed the correct process and visited an Apple Authorized Service Center. The issue was officially recorded in their service report and even captured on video. However, after keeping the phone for two days, the service center verbally denied the issue and refused to issue a defective certificate, which Amazon requires for replacement.

Because of this, I am now running out of my replacement window, despite the product being defective from the beginning. I had to take leave from work, visit the service center multiple times, and still ended up helpless.

This is a premium flagship device, and such a defect on a brand new phone is unacceptable. Even more disappointing is how difficult it has been to get a fair replacement after following all official steps.

I expected much better quality checks and post-sale support for a product of this price.

Buyers beware: Please check your device very carefully immediately after delivery and be prepared for a long struggle if you receive a defective unit."""
result = st_model.invoke(review)

# print(type(result)) #<class 'dict'>

print(result) #no need to type result.content
# print(result.keys()) # all features of output