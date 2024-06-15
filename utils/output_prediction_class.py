"""
Module for classifying and generating text using a pre-trained model.
"""

import os
import joblib
from together import Together


class Output:
    """
    A class used to classify and generate text.

    Attributes
    ----------
    model : object
        The pre-trained text classifier model.
    client : object
        The client to communicate with the Together API.

    Methods
    -------
    classify_text(text)
        Classifies the given text and returns a label indicating its quality.
    generate_text(prompt)
        Generates text based on the given prompt and classifies it.
    """

    def __init__(self):
        """
        Initializes the Output class by loading the model and setting up the API client.
        """
        self.model = joblib.load("models/best_text_classifier.joblib")
        self.client = Together(api_key=os.environ.get("TOGETHER_AI_KEY"))

    def classify_text(self, text):
        """
        Classifies the given text and returns a label indicating its quality.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        str
            The classification label of the text.
        """
        print(text)
        classification_label = self.model.predict([text])[0]

        if classification_label == 0:
            return "The output contains factual errors."
        if classification_label == 1:
            return "The output is too broad."
        return "The output is high quality."

    def generate_text(self, prompt):
        """
        Generates text based on the given prompt and classifies it.

        Parameters
        ----------
        prompt : str
            The prompt to generate text from.

        Returns
        -------
        dict
            A dictionary containing the generated text and its classification label.
        """
        response = self.client.chat.completions.create(
            model="bamartin1618@gmail.com/OpenHermes-2p5-Mistral-7B-xi-jinping-chatbot-2024-05-11-16-25-08-c0bca36d",
            messages=[{"role": "user", "content": prompt}],
        )
        label = self.classify_text(response.choices[0].message.content)
        return {"text": response.choices[0].message.content, "label": label}
