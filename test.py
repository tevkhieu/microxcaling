from transformers import pipeline

pipe = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)

# Clear entailment: a dog is an animal
print(pipe("A dog is playing in the park. A dog is an animal."))

# Clear contradiction: sky is blue vs sky is green  
print(pipe("The sky is blue. The sky is green."))

print(pipe.model)