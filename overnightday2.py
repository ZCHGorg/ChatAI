import random
import string
import time
import re
import copy
#import openai
import sys
import nltk
from nltk.corpus import words, brown
nltk.download('punkt')
import tempfile
import Levenshtein
import collections
import threading
from threading import Thread, Lock
import requests
import numpy as np
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import termcolor
from termcolor import colored  # Install 'termcolor' package for colored text
import hashlib
# import numba.rocm as roc
# import numba.cuda as cuda   DON'T FORGET TO ENABLE THESE AND THE METHOD FOR THESE WHEN YOU ENABLE GPU

def create_scratch_drive():
    with tempfile.NamedTemporaryFile() as f:
        return f.name

scratch_drive = create_scratch_drive()

def upload_training_data(file_name, directory):
    with open(file_name, "rb") as file:
        data = file.read()

    response = requests.post(f"http://localhost:8080/upload/{directory}", data=data)

    if response.status_code == 200:
        return True
    else:
        return False

def download_training_data(file_name, directory):
    response = requests.get(f"http://localhost:8080/download/{directory}/{file_name}")

    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False
    
def scrape_web_page(url):
    with threading.RLock() as scrape_web_page_lock:
        response = requests.get(url)

        if response.status_code == 200:
            return response.text
        else:
            return None

# Download NLTK words dataset if not already downloaded
nltk.download('words', download_dir="./")
nltk.download('brown', download_dir="./")

class SelfImprovingBot:
    shared_context_history = None  # Initialize the class attribute
    
    def __init__(self, decay_factor=0.95, max_context_length=5000, dynamic_context_window=550, name="name"):
        self.max_context_length = max_context_length
        self.name = name
        self.dynamic_context_window = dynamic_context_window
        self.context_history = collections.deque(maxlen=max_context_length)
        self.user_feedback = {}
        self.external_knowledge_map = {}
        self.learning_rate = 0.7
        self.response_quality = {}
        self.self_code_improvement = True
        self.code_improvement_strategy = "context_aware"
        self.ml_model = True
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
        self.scrape_web_page_lock = multiprocessing.Lock()
        self.context_history_for_current_topic_lock = multiprocessing.Lock()
        self.simulate_conversation_lock = multiprocessing.Lock()  # Add a lock for concurrent access
        self.context_file_path = f"{name}_context.txt"
        self.decay_factor = decay_factor  # Decay factor to reduce the weight of scores over time


    def update_conversation_history(self, new_context_history):
        with self.context_history_lock:
            self.context_history.append(new_context_history)

            if len(self.context_history) > self.max_context_length:
                self.context_history.pop(0)

            # Convert the context history to a multi-dimensional array
            context_history_array = np.array(self.context_history)

            # Get the current topic of the conversation
            current_topic = context_history_array[:, 0]

            # Update the context history for the current topic
            with self.context_history_for_current_topic_lock:
                context_history_for_current_topic = context_history_array[context_history_array[:, 0] == current_topic]
                self.context_history_for_current_topic = context_history_for_current_topic[-self.max_context_length:]


    # def query_chatgpt(self, prompt):
    #     if time.time() - self.last_query_time < 3600:  # 3600 seconds = 1 hour
    #         print("Rate limit exceeded. Wait for an hour.")
    #         return ""

    #     openai.api_key = 'GIVE_ME_YOUR_API'

    #     response = openai.Completion.create(
    #         engine="text-davinci-003",
    #         prompt="You are a helpful assistant that provides solutions to evolution challenges.\nUser: " + prompt,
    #         max_tokens=150,
    #         temperature=0.7
    #     )

    #     self.last_query_time = time.time()
    #     return response.choices[0].text.strip()

    def process_user_input(self, user_input, lang="en"):
        response = self.generate_response(user_input, lang)
        return response
    
    def generate_response(self, user_input, lang="en"):
            # Convert context history to a list for slicing
            context_list = list(self.context_history)
            context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
            
            context = tuple(context_subset)  # Convert the subset back to a tuple for 'context'

            if context in self.response_cache:
                return self.response_cache[context]

            if context in self.context_history:
                response = random.choice(self.context_history[context])
            else:
                # Select a random sentence from the brown corpus
                random_sentence = random.choice(brown.sents())
                random_sentence = ' '.join(random_sentence)
                response = f"Time organizes randomness. {random_sentence}" 

            if self.self_code_improvement:
                response = self.improve_own_code(response, context)

            self.response_cache[context] = response
            return response

    def self_improve(self):
        new_context_history = []

        # Call the optimized_self_improve method and retrieve any context history to share
        optimized_new_context_history = self.optimized_self_improve()
        new_context_history.extend(optimized_new_context_history)

        # Continue with the remaining self-improvement steps
        self.update_shared_context_history(new_context_history)
        self.update_context_history()
        self.update_learning_rate()
        self.analyze_response_quality()

        if self.ml_model is True:
            self.ml_model = self.train_ml_model()

        self.generate_feedback()
        self.learn_from_self()
        self.optimize_resources()

        # if self.code_improvement_strategy == "context_aware":
        #     self.deterministic_fallback()

    def update_shared_context_history(self, new_context_history):
        with self.context_history_lock:
            self.shared_context_history.extend(new_context_history)

    def optimized_self_improve(self):
        current_time = time.time()
        if current_time - self.last_self_improve_time >= 7200:  # 7200 seconds = 2 hours
            # Implement optimized self-improvement mechanism here
            self.last_self_improve_time = current_time

            # Generate and return context history entries based on your logic
            generated_context_entries = [
                "Generated context entry 1",
                "Generated context entry 2",
                # ... add more entries based on your logic
            ]
            
            return generated_context_entries
        
        return []  # Return an empty list if no entries are generated


    def update_context_history(self):
        with self.context_history_lock:
            new_context_history = self.generate_response("continue to improve to improve my own code and resource usage with each iteration by talking to myself")
            self.context_history.append(new_context_history)

            if len(self.context_history) > self.max_context_length:
                self.context_history.popleft()

    def get_context_history(self):
        with self.context_history_lock:
            return list(self.context_history)

    def get_changes_to_context_history(self):
        with self.context_history_lock:
            changes_to_context_history = []
            for i in range(len(self.context_history) - 1, -1, -1):
                if self.context_history[i] != self.context_history[i - 1]:
                    changes_to_context_history.append(self.context_history[i])
            return changes_to_context_history

    # def deterministic_fallback(self):
    #     improved_bot = self.code_versions[-1]
    #     current_bot = self
    #     if current_bot.performance_degraded(improved_bot):
    #         self = improved_bot
    #         print("Fallback: Performance degraded. Rolled back to previous version.")

    #     # Get the most recent context
    #     context = self.context_history[-1]

    #     # Generate a response based on the context
    #     response = self.generate_response(context)

    #     return response

    # def performance_degraded(self, improved_bot):
    #     return random.random() < 0.01

    def update_learning_rate(self):
        if self.foreground_accuracy_history:
            foreground_average_accuracy = sum(self.foreground_accuracy_history) / len(self.foreground_accuracy_history)
            color_indicator = "No Change"  # Default indicator for no significant change
            
            if self.last_foreground_average_accuracy is not None:
                if foreground_average_accuracy > self.last_foreground_average_accuracy:
                    color_indicator = colored("Green", "green")  # Colored output for accuracy improvement
                    self.learning_rate = min(0.7, self.learning_rate * 1.1)
                elif foreground_average_accuracy < self.last_foreground_average_accuracy:
                    color_indicator = colored("Red", "red")  # Colored output for accuracy decline
                    self.learning_rate = max(0.1, self.learning_rate / 1.1)
                
            # Make sure the learning rate stays within the desired range
            self.learning_rate = max(0.1, min(0.7, self.learning_rate))
            
            self.last_foreground_average_accuracy = foreground_average_accuracy
            
            print(f"Accuracy Color: {color_indicator} | Learning Rate: {self.learning_rate:.3f}")
        else:
            print("No accuracy (yet)")

    def analyze_response_quality(self):
        with self.context_history_lock:
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
            if "[External Info]" in current_response:
                improved_response = current_response.replace("[External Info]", "[Additional Information]")
            else:
                improved_response = self.apply_regular_expression(current_response)  # Remove accuracy from here
                improved_response = self.apply_ml_suggestions(improved_response, context)

            # Pass only the response text to the predict_second_sentence function
            second_hidden_sentence = self.predict_second_sentence(improved_response)
            accuracy = self.compare_sentences(improved_response, second_hidden_sentence)

            # Update accuracy history based on the code improvement strategy
            if self.self_code_improvement:
                self.foreground_accuracy_history.append(accuracy)
            # else:
            #     self.background_accuracy_history.append(accuracy)

            # Print overall accuracy drift
            self.print_accuracy_drift(bot_name)

            return improved_response, accuracy
        
    def generate_unique_color(self, bot_name):
        hash_value = int(hashlib.sha256(bot_name.encode()).hexdigest(), 16)
        color_list = [
            "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
            "light_grey", "dark_grey", "light_red", "light_green", "light_yellow",
            "light_blue", "light_magenta", "light_cyan"
        ]
        color_index = hash_value % len(color_list)
        bot_color = color_list[color_index]
        return bot_color

    def print_accuracy_drift(self, bot_name):
        num_iterations_for_change = 10  # Number of iterations to consider for average accuracy change

        if len(self.foreground_accuracy_history) >= num_iterations_for_change:
            last_n_avg_accuracy = sum(self.foreground_accuracy_history[-num_iterations_for_change:]) / num_iterations_for_change

            if self.last_foreground_average_accuracy is not None:
                if last_n_avg_accuracy > self.last_foreground_average_accuracy:
                    change_color = "green"
                    self.accuracy_change_count += 1  # Increment the accuracy change counter
                elif last_n_avg_accuracy < self.last_foreground_average_accuracy:
                    change_color = "red"
                    self.accuracy_change_count += 1  # Increment the accuracy change counter
                else:
                    change_color = "blue"  # Display "No Change" in blue

                bot_color = self.generate_unique_color(bot_name)  # Generate a unique color for the bot

                change_text = "Improving" if change_color == "green" else ("Declining" if change_color == "red" else "No Change")
                change_text_colored = colored(change_text, change_color)
                bot_name_colored = colored(bot_name, bot_color, attrs=["bold"])

                print(f"{bot_name_colored}, AVG.Accuracy: {last_n_avg_accuracy:.4f}, Change: {change_text_colored}, Count: {self.accuracy_change_count}")
                self.last_foreground_average_accuracy = last_n_avg_accuracy
        else:
            print(f"{bot_name}, Not enough iterations for accuracy comparison.")

    def print_global_accuracy(self, bot_name):
        num_iterations_for_change = 10  # Number of iterations to consider for average accuracy change

        if len(self.foreground_accuracy_history) >= num_iterations_for_change:
            last_n_avg_accuracy = sum(self.foreground_accuracy_history[-num_iterations_for_change:]) / num_iterations_for_change

            if self.last_global_average_accuracy is not None:
                if last_n_avg_accuracy > self.last_global_average_accuracy:
                    global_change_color = "green"
                elif last_n_avg_accuracy < self.last_global_average_accuracy:
                    global_change_color = "red"
                else:
                    global_change_color = None

                if global_change_color:
                    global_change_text = colored(f"{last_n_avg_accuracy:.4f}", global_change_color)  # Highlight the accuracy score
                else:
                    global_change_text = f"{last_n_avg_accuracy:.4f}"

                print(f"Bot: {bot_name}, DO YOU SEE ME NOW??????????????????????????????????????????????????????????????????????? Global AVG.Accuracy: {global_change_text}")
                self.last_global_average_accuracy = last_n_avg_accuracy
        else:
            print("Not enough iterations for global accuracy comparison.")
        
    def predict_second_sentence(self, response_text):
        # Tokenize the response text
        words = nltk.word_tokenize(response_text)

        # Calculate the most common words in the tokenized response
        most_common_words = nltk.FreqDist(words).most_common(5)

        # Join the most common words to create the second sentence
        second_sentence = " ".join([word[0] for word in most_common_words])

        return second_sentence

    def apply_ml_suggestions(self, response, context):
        if self.ml_model:
            response_text = response.lower().strip()
            
            context_suggestions = [suggestion for suggestion in self.context_history if response_text not in suggestion]

            if context_suggestions:
                print("MLContext Suggestions:", context_suggestions)

                relevance_scores = self.calculate_similarity(response_text, context_suggestions)

                # Apply the decay factor to the relevance scores
                decayed_scores = [score * self.decay_factor ** idx for idx, score in enumerate(relevance_scores)]

                most_relevant_index = np.argmax(decayed_scores)
                most_relevant_suggestion = context_suggestions[most_relevant_index]

                if isinstance(most_relevant_suggestion, tuple):
                    most_relevant_suggestion = most_relevant_suggestion[0]

                improved_response = most_relevant_suggestion
                return improved_response

        return response
    
    def calculate_similarity(self, response_text, context_suggestions):
        vectorizer = TfidfVectorizer()
        all_texts = [response_text] + [suggestion[0] for suggestion in context_suggestions]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        relevance_scores = similarity_matrix[0, 1:]
        return relevance_scores
    ######## REMOVE COMMENTS FROM THE FOLLOWING AND COMMENT OUT THE ABOVE WHEN ENABLE GPU #######
    # def calculate_similarity(self, response_text, context_suggestions):
    #     vectorizer = TfidfVectorizer()
    #     all_texts = [response_text] + [suggestion[0] for suggestion in context_suggestions]
    #     tfidf_matrix = vectorizer.fit_transform(all_texts)
        
    #     if cuda.is_available():
    #         with cuda.gpus[0]:
    #             similarity_matrix = cosine_similarity(tfidf_matrix)
    #     elif roc.is_available():
    #         with roc.gpus[0]:
    #             similarity_matrix = cosine_similarity(tfidf_matrix)
    #     else:
    #         similarity_matrix = cosine_similarity(tfidf_matrix)

    #     relevance_scores = similarity_matrix[0, 1:]
    #     return relevance_scores

    def compare_sentences(self, sentence1, sentence2):
        # This function compares two sentences and returns the accuracy of the prediction.
        levenshtein_distance = Levenshtein.distance(sentence1, sentence2)
        accuracy = 1 - (levenshtein_distance / max(len(sentence1), len(sentence2)))
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

    def generate_feedback(self):
        for response in self.context_history:
            if self.response_quality.get(response, 0) > 0.6:
                feedback = f"Bot's response '{response}' was useful."
                if feedback not in self.user_feedback:
                    self.user_feedback[feedback] = 1
                else:
                    self.user_feedback[feedback] += 1

    def learn_from_self(self):
        self_improvement_context = random.choice(self.context_history)
        self_improvement_dialogue = " ".join([x for x in self_improvement_context if isinstance(x, str)])
        improved_code = self.improve_own_code(self_improvement_dialogue, self_improvement_context)
        self.context_history[self.context_history.index(self_improvement_context)] = improved_code

    def optimize_resources(self):
        if self.model_size > self.memory_threshold:
            self.compress_model()

    def compress_model(self):
        self.model_size /= 2

    def retrieve_external_knowledge(self, state):
        with self.scrape_web_page_lock:
            knowledge = "External Knowledge for State: " + str(state)
            return knowledge

    def process_user_input(self, user_input, lang="en"):
        response = self.generate_response(user_input, lang)
        return response

    # def improve_own_knowledge(self):
    #     with self.context_history_lock:
    #         # Convert context history to a list for slicing
    #         context_list = list(self.context_history)
    #         context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
            
    #         state = tuple(context_subset)  # Convert the subset back to a tuple for the 'state'

    #         external_info = self.retrieve_external_knowledge(state)
            
    #         if external_info:
    #             print("Running simulate_conversation to generate external information...")
    #             self.simulate_conversation()  # Run simulate_conversation to get external information
                
    #             # Rest of your code for processing the simulated conversation output
    #             # ...

    def improve_own_knowledge(self, simulate_conversation_call=True):
        if not simulate_conversation_call:
            # Convert context history to a list for slicing
            context_list = list(self.context_history)
            context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
            
            state = tuple(context_subset)  # Convert the subset back to a tuple for the 'state'

            external_info = self.retrieve_external_knowledge(state)
            
            if external_info:
                print("RandSentence")
                new_letter = str(random.choice(brown.sents()))
                #random_sentence = ' '.join(random_sentence)
            #  new_letter = random.choice(string.ascii_letters)  # Generate a random letter
                self.memory += new_letter  # Inject the new sentence into the memory

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

    def simulate_conversation(self):
        with self.simulate_conversation_lock:
            while True:
                user_input = "hello Generate_response."
                response = self.generate_response(user_input)
                print(response)
                conversation = self.generate_random_conversation()  # Generate a random conversation

                for user_input, lang in conversation:
                    random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
                    sentences = nltk.sent_tokenize(random_paragraph)

                    while len(sentences) < 4:
                        print("Not enough sentences in the paragraph to extract two sentences.")
                        self.improve_own_knowledge(simulate_conversation_call=False)  # Skip the improvement logic
                        conversation = self.generate_random_conversation()  # Generate a new conversation
                        random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
                        sentences = nltk.sent_tokenize(random_paragraph)

                    sentence1 = sentences[0]
                    sentence2 = sentences[1]
                    user_input = str(f"(SimInput-response) {sentence1} {sentence2}")
                    print("SIM User Input:", user_input)

                    sentence3 = sentences[2]
                    sentence4 = sentences[3]
                    response = str(f"(SIMgen-response). {sentence3} {sentence4}")
                    print("Sim Bot response:", response)

                    time.sleep(random.randint(1, 2))
                    improved_response, accuracy = self.improve_own_code(response, user_input)
                    print(f"Impr. response: {improved_response} (Accuracy: {accuracy})")

                    # Rest of your code for updating context_history, self-improvement, etc.
                    self.context_history = [user_input, response, improved_response]
                    self.improve_own_knowledge()
                    self.optimize_resources()
                    self.self_improve()

    def generate_random_conversation(self):
        num_turns = random.randint(3, 10)  # Generate a random number of conversation turns
        conversation = []

        for _ in range(num_turns):
            user_input = "User input " + str(_)
            lang = "en"
            conversation.append((user_input, lang))

        return conversation

def bot_process(bot_name):
    bot = SelfImprovingBot(max_context_length=5000, dynamic_context_window=550, name=bot_name)
    print(f"{bot_name} is ready.")
    
    while True:
        bot.simulate_conversation()
        time.sleep(random.randint(1, 2))  # Add some delay between conversations


if __name__ == "__main__":
    num_bots = 10
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