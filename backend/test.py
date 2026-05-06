"""Test retrieval""" 
# from retrieval import retrieve

# results = retrieve("How do I solve a quadratic equation by factoring?")
# for r in results:
#     print(f"Score: {r['score']:.3f} | Topic: {r['metadata'].get('topic')}")
#     print(r['text'][:200])
#     print("---")



"""Test Groq"""
# from groq import Groq
# from config import settings


# client = Groq(api_key=settings.groq_api_key)
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain how to solve a linear equation",
#         }
#     ],
#     model=settings.groq_model,
#     max_tokens=50,
# )

# print(chat_completion.choices[0].message.content)

"""Test Langfuse"""

from observe import start_trace, finish_trace

trace = start_trace("test", {"query": "hello"})
finish_trace(trace, "world")

# from langfuse import Langfuse
# from config import settings

# lf = Langfuse(
#     secret_key=settings.langfuse_secret_key,
#     public_key=settings.langfuse_public_key,
#     host=settings.langfuse_base_url,
# )

# # See what methods are available
# methods = [m for m in dir(lf) if not m.startswith('_')]
# print(methods)
