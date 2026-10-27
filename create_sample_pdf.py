"""
create_sample_pdf.py - Generates an educational sample PDF about AI & Machine Learning.

This script creates 'sample_ai_overview.pdf' with 4 structured pages, perfect for
testing and demonstrating the AI PDF Question-Answering Assistant.
"""

import pymupdf


def generate_sample_pdf(output_filename: str = "sample_ai_overview.pdf") -> str:
    """Creates a 4-page educational PDF on AI & Machine Learning."""
    doc = pymupdf.open()

    pages_content = [
        # Page 1
        (
            "Page 1: Introduction to Artificial Intelligence & Machine Learning",
            [
                ("What is Artificial Intelligence?",
                 "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
                 "that are programmed to think and learn. The core goal of AI is to perform tasks such as "
                 "visual perception, speech recognition, decision-making, and language translation."),

                ("Types of Machine Learning",
                 "Machine Learning (ML) is a subfield of AI focused on building systems that learn from data. "
                 "There are three primary paradigms: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. "
                 "In Supervised Learning, models train on labeled datasets containing inputs and known outputs. "
                 "Examples include Linear Regression, Logistic Regression, and Support Vector Machines (SVM). "
                 "In Unsupervised Learning, algorithms discover hidden patterns or clusters in unlabeled data, "
                 "such as K-Means clustering and Principal Component Analysis (PCA). "
                 "In Reinforcement Learning, an agent learns to make optimal decisions by interacting with an "
                 "environment through trial-and-error, guided by rewards and penalties.")
            ]
        ),
        # Page 2
        (
            "Page 2: Deep Learning & Neural Network Architectures",
            [
                ("Deep Learning Fundamentals",
                 "Deep Learning is a subset of machine learning based on artificial neural networks with multiple "
                 "hidden layers. Inspired by the human brain, deep neural networks can automatically learn hierarchical "
                 "representations of complex data such as images, audio, and raw text."),

                ("Activation Functions and Backpropagation",
                 "Neural networks introduce non-linearity through activation functions. The Rectified Linear Unit (ReLU) "
                 "is widely used because it accelerates convergence and mitigates the vanishing gradient problem. "
                 "Sigmoid and Tanh functions squish activations between 0 to 1 and -1 to 1 respectively. "
                 "Training neural networks relies on Backpropagation, an efficient algorithm that computes the gradient "
                 "of the loss function with respect to each network weight using the chain rule of calculus. "
                 "Optimization algorithms like Stochastic Gradient Descent (SGD) and Adam iteratively update the weights "
                 "to minimize loss.")
            ]
        ),
        # Page 3
        (
            "Page 3: Natural Language Processing & Information Retrieval",
            [
                ("Natural Language Processing (NLP)",
                 "Natural Language Processing (NLP) enables computers to understand, interpret, and manipulate human language. "
                 "Key preprocessing steps include tokenization, lowercase normalization, stop-word removal, and lemmatization."),

                ("TF-IDF and Cosine Similarity in Document Search",
                 "TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates the importance of a word "
                 "within a document relative to a collection of documents (corpus). "
                 "Term Frequency measures how frequently a word occurs in a specific passage. "
                 "Inverse Document Frequency scales down common words (like 'the', 'is') and elevates informative terms. "
                 "Cosine Similarity calculates the cosine of the angle between two multi-dimensional vectors. "
                 "When a user asks a question, both the question and document passages are transformed into TF-IDF vectors. "
                 "The cosine similarity metric produces a score between 0.0 and 1.0, identifying the most relevant passage "
                 "independent of passage length.")
            ]
        ),
        # Page 4
        (
            "Page 4: Model Evaluation and Overfitting Prevention",
            [
                ("Evaluation Metrics for Classification",
                 "Accuracy measures the proportion of correct predictions out of all predictions. However, when classes "
                 "are imbalanced, accuracy can be misleading. "
                 "Precision calculates the proportion of true positives among all predicted positives, answering: of all predicted "
                 "positives, how many were truly positive? "
                 "Recall measures the ability of a model to identify all relevant instances, answering: of all actual positives, "
                 "how many did the model catch? "
                 "The F1 Score provides the harmonic mean of precision and recall, balancing both metrics."),

                ("Overfitting and Regularization Techniques",
                 "Overfitting occurs when a machine learning model memorizes training noise and fails to generalize to unseen test data. "
                 "Standard techniques to prevent overfitting include K-Fold Cross-Validation, L1 (Lasso) and L2 (Ridge) Regularization, "
                 "Dropout in neural networks, and Early Stopping during model training.")
            ]
        )
    ]

    for title, sections in pages_content:
        page = doc.new_page(width=595, height=842)  # Standard A4 size in points
        # Insert Header Title
        page.insert_text((50, 60), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.5))

        y_offset = 100
        for sec_title, body_text in sections:
            # Insert section heading
            page.insert_text((50, y_offset), sec_title, fontsize=12, fontname="helv", color=(0.2, 0.2, 0.2))
            y_offset += 20

            # Insert body text inside a bounded rectangle to wrap automatically
            rect = pymupdf.Rect(50, y_offset, 545, y_offset + 250)
            # PyMuPDF insert_textbox wraps text cleanly
            page.insert_textbox(rect, body_text, fontsize=10, fontname="times-roman", lineheight=1.4)
            y_offset += 160

        # Footer
        page.insert_text((50, 810), "AI PDF Assistant - Sample Study Guide for B.Tech AI/ML", fontsize=8, color=(0.5, 0.5, 0.5))

    doc.save(output_filename)
    doc.close()
    print(f"Sample PDF successfully generated: {output_filename}")
    return output_filename


if __name__ == "__main__":
    generate_sample_pdf()
