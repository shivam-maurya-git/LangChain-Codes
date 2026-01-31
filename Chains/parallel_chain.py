from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableParallel # Helps in executing multiple chains parallely

load_dotenv()

models = ["DeepHat/DeepHat-V1-7B","deepseek-ai/DeepSeek-R1","meta-llama/Llama-3.1-8B-Instruct"]
model1 = ChatHuggingFace(llm = HuggingFaceEndpoint(repo_id=models[0],
                             task='text-generaation'))
model2 = ChatHuggingFace(llm = HuggingFaceEndpoint(repo_id=models[1],
                             task='text-generaation'))
model3 = ChatHuggingFace(llm = HuggingFaceEndpoint(repo_id=models[2],
                             task='text-generaation'))
notes = "A nebula is an interstellar cloud of gas and dust. The properties of nebulae vary enormously and depend on their composition as well as the environment in which they are situated. Emission nebulae are powered by young, massive stars and emit their own light, reflection nebulae shine by reflecting light from nearby massive stars, and dark nebulae, as the name suggests, are dark and can only be seen when silhouetted against a bright background. Nebulae can also result from the end stages of stellar evolution. In this case they are present as either a planetary nebula or a supernova remnant depending on the mass of the dying star. Dark nebulae are interstellar clouds that contain a very high concentration of dust. This allows them to scatter and absorb all incident optical light, making them completely opaque at visible wavelengths. They are most obvious when located in front of a bright emission nebula (e.g. the Horsehead nebula in Orion) or in a region that is very rich in stars (e.g. Barnard 68 in Ophiuchus). The most famous example in the southern hemisphere is the Coal Sack nebula near α Crux in the Southern Cross.The average temperature inside a dark nebula ranges from about 10 to 100 Kelvin, allowing hydrogen molecules to form and star formation to take place. Large dark nebulae that can contain over a million solar masses of material and extend over 200 parsecs are known as giant molecular clouds. The smallest ones, called Bok globules, tend to be less than 3 light years across and contain less than 2000 solar masses of material."
prompt1  = PromptTemplate(
    template="Generate short notes on the {notes}",
    input_variables=['notes']
)

prompt2 = PromptTemplate(
    template='Generate four short questions on the following {notes}',
    input_variables={'notes'}
)
#both prompt 1 and 2 are working on same notes (input)
prompt3 = PromptTemplate(
    template='Merge the provides notes and questions in single document. Notes : {notes} and Questions : {quiz}',
    input_variables=['notes','quiz']
)
parser  = StrOutputParser()
parallel_chain = RunnableParallel(
    {'notes': prompt1 | model1 | parser,
     "quiz":prompt2| model2|parser}
)
# Need to chain same names as input variables of prompt 3, so that third model can treat each chain as one input easily.
merge_chain = prompt3 | model3| parser
# parallel_chain is already giving parsered output

final_chain = parallel_chain | merge_chain

result = final_chain.invoke({'notes' : notes})
# Based on notes input notes will be created and quiz also, that will be given to model3 for the mergeing

# print(result)

final_chain.get_graph().print_ascii()

