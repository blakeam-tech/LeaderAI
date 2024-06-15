"""
This module provides a service class for interacting with Firestore.
"""

import firebase_admin
from firebase_admin import credentials, firestore


class FirestoreService:
    """
    Service class to interact with Firestore.
    """

    def __init__(self, credential_path):
        """
        Initialize the Firestore service with credentials from a given file.

        Args:
            credential_path (str): Path to the Firebase Admin SDK JSON key file.
        """
        # Check if any Firebase apps have been initialized yet; if not, initialize one
        if not firebase_admin._apps:
            self.cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(self.cred)

        # Obtain a client instance for the Firestore service
        self.firestore_client = firestore.client()

    def add_document(self, collection_name, document_data):
        """
        Adds a document to a specified Firestore collection and returns the new document reference.

        Args:
            collection_name (str): The name of the Firestore collection where the document will be added.
            document_data (dict): Data to be stored in the document.

        Returns:
            firestore.DocumentReference: Reference to the newly added document.
        """
        # Get a reference to the specified Firestore collection
        collection_ref = self.firestore_client.collection(collection_name)

        # Add a new document to the collection
        doc_ref, _ = collection_ref.add(document_data)

        # Return the document reference
        return doc_ref

    def get_document(self, collection_name, document_id):
        """
        Retrieves a document from a specified Firestore collection by document ID.

        Args:
            collection_name (str): The name of the Firestore collection.
            document_id (str): The ID of the document to retrieve.

        Returns:
            dict: The document data if found, otherwise None.
        """
        # Get a reference to the specified Firestore collection
        doc_ref = self.firestore_client.collection(collection_name).document(document_id)

        # Retrieve the document
        doc = doc_ref.get()

        # Return the document data if the document exists
        if doc.exists:
            return doc.to_dict()
        else:
            return None
        