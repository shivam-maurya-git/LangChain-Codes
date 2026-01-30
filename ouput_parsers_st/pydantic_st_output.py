from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
class Product_review(BaseModel):
    platform : str = "amazon"  #Default value
    sentiment : Literal["Positive","Negative","Neutral"]
    # email: EmailStr  #built in email valider
    # cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student') #gt : greater than

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
st_model = model.with_structured_output(Product_review)
review = """I purchased a brand new iPhone 17 Pro Max from Amazon, and within days of delivery the device started showing a visible red/dead pixel issue with intermittent flickering. The defect is clearly visible to the naked eye and appears across multiple screens and colors.

I immediately followed the correct process and visited an Apple Authorized Service Center. The issue was officially recorded in their service report and even captured on video. However, after keeping the phone for two days, the service center verbally denied the issue and refused to issue a defective certificate, which Amazon requires for replacement.

Because of this, I am now running out of my replacement window, despite the product being defective from the beginning. I had to take leave from work, visit the service center multiple times, and still ended up helpless.

This is a premium flagship device, and such a defect on a brand new phone is unacceptable. Even more disappointing is how difficult it has been to get a fair replacement after following all official steps.

I expected much better quality checks and post-sale support for a product of this price.

Buyers beware: Please check your device very carefully immediately after delivery and be prepared for a long struggle if you receive a defective unit."""
result = st_model.invoke(review)
# print(result)

# print(type(result)) #<class '__main__.Product_review'>
# so, we need convert it into dict

result_dict = dict(result) # converting pydantic object into dict

print(result_dict["sentiment"])

