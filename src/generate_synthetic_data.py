import pandas as pd
import config
from sklearn.model_selection import train_test_split

def generate_data():
    data = []
    
    # 1. In-domain Factual
    factual_queries = [
        "What is the semester fee deadline?",
        "When is the last date to pay fees for the odd semester?",
        "What is the penalty for late fee payment?",
        "How much is the late fee?",
        "What is the eligibility for undergraduate admissions?",
        "What marks do I need in 12th for UG admission?",
        "What is the minimum attendance required?",
        "Can I get medical leave for attendance?",
        "What is the curfew time in the hostel?",
        "Is alcohol allowed in the hostel?"
    ] * 10
    for q in factual_queries:
        data.append({"query": q, "label": "in-domain-factual"})
        
    # 2. In-domain Procedural
    procedural_queries = [
        "How do I apply for a fee waiver?",
        "What is the procedure to get a fee waiver?",
        "What documents do I need to submit for fee waiver?",
        "How do I apply for a hostel room?",
        "What is the process for hostel room allocation?",
        "How can I get permission to leave the hostel after curfew?",
        "What are the steps to submit a medical certificate?",
        "How do I pay my semester fees online?",
        "Where do I submit the income certificate?",
        "What is the admission procedure for postgraduate programs?"
    ] * 10
    for q in procedural_queries:
        data.append({"query": q, "label": "in-domain-procedural"})
        
    # 3. Out-of-domain
    ood_queries = [
        "Who is the Prime Minister of India?",
        "What is the capital of France?",
        "How do I make a chocolate cake?",
        "Write a python script for web scraping.",
        "What is quantum entanglement?",
        "Who won the FIFA World Cup in 2022?",
        "Tell me a joke.",
        "What is the weather today?",
        "How do I invest in stocks?",
        "What are the best movies of 2023?"
    ] * 10
    for q in ood_queries:
        data.append({"query": q, "label": "out-of-domain"})
        
    df = pd.DataFrame(data)
    
    # Split 80/20
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    print(f"Generated {len(train_df)} training and {len(test_df)} testing queries.")
    print(f"Saved to {config.TRAIN_CSV} and {config.TEST_CSV}")

if __name__ == "__main__":
    generate_data()
