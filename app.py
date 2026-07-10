import gradio as gr
from core.rag_pipeline import AdvancedRAGPipeline

# Initialize RAG Pipeline (using our existing architecture)
rag_pipeline = AdvancedRAGPipeline()

def respond(message, history):
    """
    Function that takes the user's message and the chat history,
    runs it through our Advanced RAG pipeline, and returns the response.
    """
    # Execute the RAG query
    result = rag_pipeline.answer_financial_query(message)
    
    # Format the response with the retrieved sources
    answer = result.get("answer", "An error occurred.")
    
    # We yield instead of return to allow Gradio to stream if configured
    return answer

# Create a sleek ChatGPT-style interface
demo = gr.ChatInterface(
    fn=respond,
    title="🏦 Mutual Fund AI Assistant",
    description="Ask me any financial questions about the ingested Mutual Funds. I use an advanced **Cross-Encoder Re-Ranking** architecture to guarantee accuracy.",
    examples=[
        "What is the expense ratio?",
        "What is the minimum investment?",
        "What are the top holdings in this fund?"
    ],
    theme=gr.themes.Soft(primary_hue="blue")
)

if __name__ == "__main__":
    # Launch for web (HuggingFace Spaces uses port 7860 by default)
    demo.launch(server_name="0.0.0.0", server_port=7860)
