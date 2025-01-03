
import collections
import threading
from threading import Thread, Lock
import requests
import numpy as np
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import termcolor
from termcolor import colored  # Install 'termcolor' package for colored text
import hashlib
# import numba.rocm as roc
# import numba.cuda as cuda   DON'T FORGET TO ENABLE THESE AND THE METHOD FOR THESE WHEN YOU ENABLE GPU
import random
import string
import time
import re
import copy
# import openai
# from sklearn.feature_extraction.text import CountVectorizer
import sys
import nltk
from nltk.corpus import words, brown
import tempfile
import Levenshtein
#import itertools
import wikipedia

def create_scratch_drive():
    with tempfile.NamedTemporaryFile() as f:
        return f.name

scratch_drive = create_scratch_drive()
# Download NLTK words dataset if not already downloaded
nltk.download('words', download_dir="./")
nltk.download('brown', download_dir="./")
nltk.download('punkt')

class SelfImprovingBot:
    shared_context_history = None  # Initialize the class attribute
    
    def __init__(self, decay_factor=0.8, max_context_length=5000, dynamic_context_window=550, name="name"):
        self.max_context_length = max_context_length
        self.name = name
        self.dynamic_context_window = dynamic_context_window
        self.context_history = collections.deque(maxlen=max_context_length)
        self.shared_context_history = collections.deque(maxlen=max_context_length)
        self.user_feedback = {}
        self.external_knowledge_map = {}
        self.learning_rate = 0.7
        self.response_quality = {}
        self.self_code_improvement = True
        self.code_improvement_strategy = "context_aware"
  #      self.ml_model = True
        self.model_size = sys.getsizeof(self)
        self.memory_threshold = 2 ** 29  # 512 MB
        self.scratch_drive_size = 2 ** 34  # 16 GB
        self.response_cache = {}
        self.code_versions = [copy.deepcopy(self)]
        self.last_query_time = 0
        self.last_self_improve_time = time.time()
        self.memory = ""
        self.global_accuracy_scores = []
        self.accuracy_history = []
        self.last_foreground_average_accuracy = None
        self.foreground_accuracy_history = self.accuracy_history
        self.accuracy_change_count = 0
        self.context_history_lock = multiprocessing.Lock()
        self.shared_context_history_lock = multiprocessing.Lock()
        self.scrape_web_page_lock = multiprocessing.Lock()
 #       self.context_history_for_current_topic_lock = multiprocessing.Lock()
        self.simulate_conversation_lock = multiprocessing.Lock()  # Add a lock for concurrent access
        self.self_improve_lock = multiprocessing.Lock()
        self.generate_response_lock = multiprocessing.Lock()   
        self.context_file_path = f"{name}_context.txt"
        self.decay_factor = decay_factor  # Decay factor to reduce the weight of scores over time
        self.shared_context_add_count = 0 
        self.accuracy_threshold_shared = 0.05  # Initial low accuracy threshold for shared context
        self.accuracy_threshold_individual = 0.05
        self.self_context = []
        self.context_history = []
        self.ml_model = None
        # self.memory = ""
        self.lang = "en"
        #self.response_cache = {}
#        super().__init__()
        self.self_learn = True
        self.self_learning_thread = None  # Initialize thread as an instance attribute
        self.foreground_accuracy_history = []
        # self.background_accuracy_history = []


#    def query_chatgpt(self, prompt):
#        if time.time() - self.last_query_time < 3600:  # 3600 seconds = 1 hour
#            print("Rate limit exceeded. Wait for an hour.")
#            return ""
#
#        openai.api_key = 'GIVE_ME_YOUR_API'

#        response = openai.Completion.create(
#            engine="text-davinci-003",
#            prompt="You are a helpful assistant that provides solutions to evolution challenges.\nUser: " + prompt,
#            max_tokens=150,
#            temperature=0.7
#        )

#        self.last_query_time = time.time()
#        return response.choices[0].text.strip()

    def generate_response(self, user_input, lang="en"):
        with self.generate_response_lock:
    #        lang = "en"  PROPOSED CHANGE
            context = tuple(self.context_history[-self.dynamic_context_window:])

            if context in self.response_cache:
                return self.response_cache[context]

            if context in self.context_history:
                response = random.choice(self.context_history[context])
            else:

                # Generate a random paragraph (a group of sentences) from the corpus
                random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))

                # Tokenize the paragraph into sentences
                sentences = nltk.sent_tokenize(random_paragraph)

                # Ensure there are at least two sentences
                if len(sentences) >= 2:
                    # Extract two consecutive sentences from the same paragraph
                    sentence1 = sentences[0]
                    sentence2 = sentences[1]
                    accuracy = self.compare_sentences(sentence1, sentence2)
                    response = f"Time organizes randomness(gen-response). Sentence1: {sentence1} Sentence2: {sentence2} Accuracy: {accuracy:.2f}"             
                else:
                    print("Not enough sentences in the paragraph to extract two sentences.")

            if self.self_code_improvement:
                response = self.improve_own_code(response, context)

            self.response_cache[context] = response
            return response

    def _self_learn(self):
            # Perform self-learning tasks
            self.optimized_self_improve()
            self.update_context_history()
            self.update_learning_rate()
            self.analyze_response_quality()

            if self.ml_model is None:
                self.ml_model = self.train_ml_model()

            self.generate_feedback()
            self.learn_from_self()
            self.optimize_resources()

            # if self.code_improvement_strategy == "context_aware":
            #     self.deterministic_fallback()

    def optimized_self_improve(self):
        current_time = time.time()
        if current_time - self.last_self_improve_time >= 7200:  # 7200 seconds = 2 hours
            # Implement optimized self-improvement mechanism here
            self.last_self_improve_time = current_time
            
    def self_improve(self):
        new_context_history = []
        with self.self_improve_lock:
            optimized_new_context_history = self.optimized_self_improve()
            new_context_history.append(optimized_new_context_history)

            # Continue with the remaining self-improvement steps
            self.update_shared_context_history(new_context_history)

            # Rest of your code
            random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))

                # Tokenize the paragraph into sentences
            sentences = nltk.sent_tokenize(random_paragraph)

                # Ensure there are at least two sentences
            if len(sentences) >= 2:
                    # Extract two consecutive sentences from the same paragraph
                sentence1 = sentences[0]
                sentence2 = sentences[1]

                accuracy = self.compare_sentences(sentence1, sentence2)
                user_input = f"Time organizes randomness(selfIMP-user_input). Sentence1: {sentence1} Sentence2: {sentence2} Accuracy: {accuracy:.2f}"

                #user_input = str(f"Time organizes randomness(SelfIMPgen-response). {sentence1} {sentence2}")
                print("SELFIMPROVEDGenerated response:", user_input)
            else:
                print("Not enough sentences in the paragraph to extract two sentences.")
                self.self_improve()            

            # Get the user's language preference
            lang = "en"
            # Perform self-improvement tasks
            self.optimized_self_improve()
            self.update_context_history()
            self.update_learning_rate()
            self.analyze_response_quality()

            if self.ml_model is None:
                    self.ml_model = self.train_ml_model()

            self.generate_feedback()
            self.learn_from_self()
            self.optimize_resources()

    def update_shared_context_history(self, new_context_history): 
        with self.shared_context_history_lock:
            self.shared_context_history.append(self.generate_response("continue to improve to improve my own code and resource usage with each iteration by talking to myself"))

            if len(self.shared_context_history) > self.max_context_length:
                self.shared_context_history.pop(0)

    def update_context_history(self):
        with self.context_history_lock:
            self.context_history.append(self.generate_response("continue to improve to improve my own code and resource usage with each iteration by talking to myself"))

            if len(self.context_history) > self.max_context_length:
                self.shared_context_history.pop(0)

    def update_learning_rate(self):
        self.learning_rate = min(0.7, 0.1 + len(self.context_history) / 1000)

    def analyze_response_quality(self):
            for response_tuple in self.context_history:
                response_text = response_tuple[0]  # Access the response text from the tuple
                response_length = len(response_text.split())
                response_quality_score = min(1.0, response_length / 20)

                if response_tuple not in self.response_quality:
                    self.response_quality[response_tuple] = response_quality_score
                else:
                    self.response_quality[response_tuple] += response_quality_score

                # Incorporate the accuracy as the knowledge score
                accuracy = response_tuple[1]
                try:
                    response_quality_score = response_quality_score * float(accuracy)
                except:
                    pass
                self.response_quality[response_tuple] = response_quality_score

    def improve_own_code(self, current_response, context):
        if self.code_improvement_strategy == "context_aware":
            # if "[External Info]" in current_response:
            #     improved_response = current_response.replace("[External Info]", "[Additional Information]")
            # else:
            improved_response = self.apply_regular_expression(current_response)  # Remove accuracy from here
            improved_response = self.apply_ml_suggestions(improved_response, context)

            # Pass only the response text to the predict_second_sentence function
            second_hidden_sentence = self.predict_second_sentence(improved_response)
            accuracy = self.compare_sentences(improved_response, second_hidden_sentence)

            # Update accuracy history based on the code improvement strategy
            if self.self_code_improvement:
                self.foreground_accuracy_history.append(accuracy)

            self.print_accuracy_drift()

            return improved_response, accuracy
        
        elif self.code_improvement_strategy == "ml_based":
            improved_response = self.apply_ml_suggestions(current_response, context)

            # Pass only the response text to the predict_second_sentence function
            second_hidden_sentence = self.predict_second_sentence(improved_response)
            accuracy = self.compare_sentences(improved_response, second_hidden_sentence)

            # Update accuracy history based on the code improvement strategy
            if self.self_code_improvement:
                self.foreground_accuracy_history.append(accuracy)

            self.print_accuracy_drift()

            return improved_response, accuracy
        else:
            return current_response, 0.0  # Return the original response and a default accuracy

    def print_accuracy_drift(self):
        if self.foreground_accuracy_history:
            foreground_average_accuracy = sum(self.foreground_accuracy_history) / len(self.foreground_accuracy_history)
            print("Avg. accuracy:", foreground_average_accuracy)
        else:
            print("No accuracy data yet.")

        # if self.background_accuracy_history:
        #     background_average_accuracy = sum(self.background_accuracy_history) / len(self.background_accuracy_history)
        #     print("Background Average Accuracy:", background_average_accuracy)
        # else:
        #     print("No background accuracy data available.")

    def predict_second_sentence(self, response_text):
        # Tokenize the response text
        words = nltk.word_tokenize(response_text)

        # Calculate the most common words in the tokenized response
        most_common_words = nltk.FreqDist(words).most_common(5)

        # Join the most common words to create the second sentence
        second_sentence = " ".join([word[0] for word in most_common_words])

        return second_sentence

    def compare_sentences(self, sentence1, sentence2):
        # This function compares two sentences and returns the accuracy of the prediction.

        levenshtein_distance = Levenshtein.distance(sentence1, sentence2)
        accuracy = 1 - (levenshtein_distance / len(sentence1))

        return accuracy

    def apply_regular_expression(self, response):
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", response):
            improved_response = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", response)
        else:
            improved_response = response

        return improved_response

    def train_ml_model(self):
        ml_model = random.randint(1, 100)
        return ml_model
    
    def apply_ml_suggestions(self, response, context):
        if self.ml_model:
            context_suggestion = [c for c in self.shared_context_history if response not in c[0]]
            if context_suggestion:
                suggested_response = self.rank_context_suggestions(response, context_suggestion)
                return suggested_response

        # If no conditions are met to modify the response, simply return it
        return response

    def rank_context_suggestions(self, response, context_suggestions):
        ranked_suggestions = []

        for suggestion in context_suggestions:
            suggested_response = suggestion[0] # + " [Context Suggestion]"
            accuracy = self.compare_sentences(response, suggested_response)
            ranked_suggestions.append((suggested_response, accuracy))

        ranked_suggestions.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy in descending order

        if ranked_suggestions:
            return ranked_suggestions[0][0]  # Return the response with the highest accuracy
        else:
            return response
    
    def calculate_similarity(self, response_text, context_suggestions):
        vectorizer = TfidfVectorizer()
        all_texts = [response_text] + [suggestion[0] for suggestion in context_suggestions]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        relevance_scores = similarity_matrix[0, 1:]

        # Find the index of the context suggestion with the highest relevance score
        max_relevance_index = relevance_scores.argmax()

        # Return the suggested response
        suggested_response = context_suggestions[max_relevance_index][0] + " [Context Suggestion]"
        return suggested_response
        
    def get_context_suggestion(self, context):
            context_suggestion = []
            for c in self.context_history:
                if context not in c and c not in context_suggestion:
                    context_suggestion.append(c)
            return context_suggestion

    def generate_feedback(self):
        for response in self.context_history:
            if self.response_quality.get(response, 0) > 0.6:
                feedback = f"Bot's response '{response}' was useful."
                print("useful!")
                if feedback not in self.user_feedback:
                    self.user_feedback[feedback] = 1
                    print("+1 FEEDBACK!")
                else:
                    self.user_feedback[feedback] += 1
                    print("NOT +1 FEEDBACK!")

    def learn_from_self(self):
        self_improvement_context = random.choice(self.context_history)
        self_improvement_dialogue = " ".join([x for x in self_improvement_context if type(x) == str])
        improved_code, accuracy = self.improve_own_code(self_improvement_dialogue, self_improvement_context)
        self.context_history[self.context_history.index(self_improvement_context)] = improved_code

    def optimize_resources(self):
        if self.model_size > self.memory_threshold:
            self.compress_model()

    def compress_model(self):
        self.model_size /= 2
        self.self_improve()

#     def improve_own_knowledge(self):
#         state = tuple(self.context_history[-self.dynamic_context_window:])
#         external_info = self.retrieve_external_knowledge(state)
        
#         if external_info:
#             new_info = external_info
#             self.memory = new_info
# #            print("new knowledge, oh boy!")
#             self.self_improve()
#         else:
# #            print("no external source")
#             self.self_improve()
#             new_letter = random.choice(string.ascii_letters)  # Generate a random letter
#             self.memory += new_letter  # Inject the new letter into the memory

    def simulate_conversation(self):
        with self.simulate_conversation_lock:
            # Get a random article title from Wikipedia
            article_title = wikipedia.random()

            # Retrieve the article object
            article = wikipedia.page(article_title)

            # Tokenize the entire article content into sentences
            sentences = nltk.sent_tokenize(article.content)

            # Process the sentences in pairs
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):  # Ensure there are at least two sentences remaining
                    sentence1 = sentences[i]
                    sentence2 = sentences[i + 1]

                    # Process sentence pair as needed
                    self.process_sentence_pair(sentence1, sentence2)
                    time.sleep(0.1)

    def process_sentence_pair(self, sentence1, sentence2):
        accuracy = 1  # Your accuracy calculation logic here

        user_input = f"Time organizes randomness(sim-input). Sentence1: {sentence1} Sentence2: {sentence2} Accuracy: {accuracy:.2f}"
        print("SIM User Input:", user_input)

        response = f"Time organizes randomness(sim-response). Sentence1: {sentence1} Sentence2: {sentence2} Accuracy: {accuracy:.2f}"
        print("Sim Bot response:", response, accuracy)

        improved_response, accuracy = self.improve_own_code(response, user_input)
        print(f"Improved response: {improved_response} (Accuracy: {accuracy})")

    def generate_random_conversation(self, initial_user_input):
        num_turns = random.randint(3, 10)  # Generate a random number of conversation turns
        conversation = [(initial_user_input, "en")]

        for _ in range(num_turns):
            user_input = "User input " + str(_)
            lang = "en"
            conversation.append((user_input, lang))

        return conversation
            
    def retrieve_external_knowledge(self, state):
        print("retrive_external_knowledge")
        knowledge = "External Knowledge for State: " + str(state)
        return knowledge

    def handle_state_change(self, new_info):
        if self.context_history:
            old_info = self.context_history[-1]
            if old_info != new_info:
                diff_info = self.find_difference_in_info(old_info, new_info)
                if diff_info:
                    self.retrieve_external_knowledge(diff_info)

    def find_difference_in_info(self, old_info, new_info):
        conceptualized_difference = self.conceptualize_difference(old_info, new_info)
        return conceptualized_difference

    def conceptualize_difference(self, old_info, new_info):
        conceptualized_difference = "Conceptualized Difference"
        return conceptualized_difference

def bot_process(bot_name):
    bot = SelfImprovingBot(max_context_length=5000, dynamic_context_window=550, name=bot_name)
    print(f"{bot_name} is ready.")
    
    while True:
        bot.simulate_conversation()
        time.sleep(random.randint(1, 2))  # Add some delay between conversations


if __name__ == "__main__":
    num_bots = 6
    bot_names = [f"zchg.org{i}" for i in range(1, num_bots + 1)]

    manager = multiprocessing.Manager()
    shared_context_history = manager.list()

    SelfImprovingBot.shared_context_history = shared_context_history  # Assign the shared context history to the class

    processes = []
    for bot_name in bot_names:
        process = multiprocessing.Process(target=bot_process, args=(bot_name,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All bots are done.")
    
    while True:
        random_sentence = random.choice(brown.sents())
        random_sentence = ' '.join(random_sentence)
        user_input = str(f"Time organizes randomness(whiletrue). {random_sentence}")
        lang = "en"
        response = str({random_sentence})

        
        #response = bot.process_user_input(user_input, lang)
        
        # Log the conversation
        with open("conversation_log.txt", "a") as log_file:
            log_file.write(f"User: {user_input}\n")
            log_file.write(f"Bot: {response}\n")
            log_file.write("=" * 40 + "\n")
        
 #       print("Bot response:", response)

        # Continuous self-improvement loop
        # bot.improve_own_knowledge()
                #bot.optimize_resources()
        #time.sleep(random.randint(1, 2))
                #time.sleep(0.1)
                #bot.self_improve()