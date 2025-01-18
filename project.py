import csv
import os
import re
import wikipediaapi as wiki_api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import customtkinter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class WikipediaReader():
    def __init__(self, dir="articles"):
        self.pages = set()
        self.article_path = os.path.join("./", dir)
        self.wiki = wiki_api.Wikipedia(
            language='en',
            extract_format=wiki_api.ExtractFormat.WIKI,
            user_agent='Mozilla/5.0')
        try:
            os.mkdir(self.article_path)
        except OSError as e:
            print(f"Error creating directory: {e}")

    def _get_page_title(self, article):
        return re.sub(r'\s+', '_', article)

    def add_article(self, article):
        try:
            page = self.wiki.page(self._get_page_title(article))
            if page.exists():
                self.pages.add(page)
                return page
        except wiki_api.exceptions.WikipediaException as e:
            print(e)

    def process(self, update=False): # Process the articles and store them in a CSV file
        csv_file_path = os.path.join(self.article_path, 'articles.csv')
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            for page in self.pages:
                filename = re.sub(r'\s+', '_', f'{page.title}')
                filename = re.sub(r'[\(\):]', '', filename)
                file_path = os.path.join(self.article_path, f'{filename}.txt')
                if update or not os.path.exists(file_path):
                    content = page.text
                    if content:
                        tokens = word_tokenize(content)
                        tokens = [re.sub(r'[^A-Za-z0-9]+', '', token).lower() for token in tokens]
                        tokens = [token for token in tokens if token not in stop_words]
                        tokens = [lemmatizer.lemmatize(token) for token in tokens]
                        processed_content = ' '.join(tokens)
                        writer.writerow({'title': page.title, 'content': processed_content})
                    else:
                        print(f'No content for {page.title}')
                else:
                    print(f'Not updating {page.title} ...')

    def crawl_pages(self, article, depth, total_number=10):  
        print(f'Crawl {total_number} :: {article}')

        if len(self.pages) >= total_number:
            return

        page = self.add_article(article)
        childs = set()

        if page:
            for child in page.links.keys():
                if len(self.pages) < total_number:
                    self.add_article(child)
                    childs.add(child)

        depth -= 1
        if depth > 0:
            for child in sorted(childs):
                if len(self.pages) < total_number:
                    self.crawl_pages(child, depth, len(self.pages))


def create_inverted_index(corpus_path, sort_by_dropdown):
    documents = []
    with open(os.path.join(corpus_path, 'articles.csv'), 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            documents.append(row['content'])

    if sort_by_dropdown.get() == "Tf-Idf":
        vectorizer = TfidfVectorizer(use_idf=False, norm=None, token_pattern=r'(?u)\b\w+\b')  # Disable IDF and normalization to get raw term frequencies
        tf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        inverted_index = defaultdict(lambda: {'total_count': 0, 'count': 0, 'rows': defaultdict(int)})
    elif sort_by_dropdown.get() == "Boolean retrieval":
        vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b')  # Binary term presence/absence
        tf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        inverted_index = defaultdict(lambda: {'total_count': 0, 'count': 0, 'rows': defaultdict(int)})
    elif sort_by_dropdown.get() == "Okapi BM25":
        vectorizer = TfidfVectorizer(use_idf=True, norm=None, token_pattern=r'(?u)\b\w+\b')  # Use IDF for BM25
        tf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        idf = vectorizer.idf_
        k1 = 1.5
        b = 0.75
        avgdl = np.mean([len(doc.split()) for doc in documents])
        inverted_index = defaultdict(lambda: {'total_count': 0, 'count': 0, 'rows': defaultdict(int)})

    for doc_idx, doc in enumerate(documents):
        feature_index = tf_matrix[doc_idx, :].nonzero()[1]
        term_frequencies_zip = zip(feature_index, tf_matrix[doc_idx, feature_index].toarray()[0])
        for idx, tf in term_frequencies_zip:
            token = feature_names[idx]
            if sort_by_dropdown.get() == "Okapi BM25":
                dl = len(doc.split())
                bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                bm25_tf *= idf[idx]

            inverted_index[token]['rows'][doc_idx + 1] = int(tf) 
            inverted_index[token]['total_count'] += int(tf)
            inverted_index[token]['count'] += int(tf)
    return inverted_index


def create_ui():
    reader = WikipediaReader()
    inverted_index = None

    def switch_frame(frame):  #switch between the "search" and "inverted index" frames
        frame1.pack_forget()
        frame2.pack_forget()
        frame.pack(pady=20, padx=30, fill="both", expand=True)
        if frame == frame2 and inverted_index is not None:
            update_inverted_index_widget()

    def search_and_update(): #search for the query and update the results
        results_text.configure(state='normal')
        results_text.delete("1.0", "end")

        reader.pages.clear()

        reader.crawl_pages(query_entry.get(), depth=0, total_number=50)
        reader.process(update=True)

        results_text.insert("end", "Titles of articles found:\n")
        for page in reader.pages:
            results_text.insert("end", f"- {page.title}\n")
        results_text.configure(state='disabled')

        nonlocal inverted_index
        inverted_index = create_inverted_index(reader.article_path, sort_by_dropdown)
        update_inverted_index_widget()

    def boolean_search(): #perform a boolean search based on the query
        if inverted_index is None:
            update_inverted_index_widget("Inverted index is not available. Please perform a search first.")
            return

        query = boolean_search_entry.get()
        tokens = word_tokenize(query.lower())
        lemmatizer = WordNetLemmatizer()

        # Remove special characters and stop words, and lemmatize tokens
        tokens = [lemmatizer.lemmatize(re.sub(r'[^A-Za-z0-9]+', '', token)) for token in tokens]

        if not tokens:
            update_inverted_index_widget("No valid tokens found in the query.")
            return

        # Initialize sets for Boolean operations
        and_set = None
        or_set = set()
        not_set = set()

        # Determine the operation based on the query
        operation = "AND"
        for token in tokens:
            if token == "and":
                operation = "AND"
            elif token == "or":
                operation = "OR"
            elif token == "not":
                operation = "NOT"
            else:
                if operation == "AND":
                    if token in inverted_index:
                        if and_set is None:
                            and_set = set(inverted_index[token]['rows'].keys())
                        else:
                            and_set &= set(inverted_index[token]['rows'].keys())
                    else:
                        and_set = set()
                elif operation == "OR":
                    if token in inverted_index:
                        or_set |= set(inverted_index[token]['rows'].keys())
                elif operation == "NOT":
                    if token in inverted_index:
                        not_set |= set(inverted_index[token]['rows'].keys())

        # Combine the results of the Boolean operations
        if and_set is not None:
            result_set = and_set
        else:
            result_set = or_set

        result_set -= not_set

        if result_set:
            result_string = ""
            for token in tokens:
                if token in inverted_index:
                    result_string += f"Token: {token}\n"
                    result_string += f"Total Count: {inverted_index[token]['count']}\n"
                    result_string += "Articles:\n"
                    for doc in result_set:
                        if doc in inverted_index[token]['rows']:
                            result_string += f"  - Document ID: {doc}, Occurrences: {inverted_index[token]['rows'][doc]}\n"
                    result_string += "\n"
        else:
            result_string = "No documents match the query."

        update_inverted_index_widget(result_string)

    def update_inverted_index_widget(content=None):
        inverted_index_widget.configure(state='normal')
        inverted_index_widget.delete("1.0", "end")
        if content:
            inverted_index_widget.insert("1.0", content)
        else:
            inverted_index_string = '\n'.join(
                f'{k}: Occurrences = {v["count"]}, Article(s) = [{", ".join(f"{row} ({count})" for row, count in v["rows"].items())}]'
                for k, v in inverted_index.items())
            inverted_index_widget.insert("1.0", inverted_index_string)
        inverted_index_widget.configure(state='disabled')

    
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")
    window = customtkinter.CTk()
    window.after(0, lambda: window.state('zoomed'))
    window.title("Wikipedia Article Search Engine")

    # Create the frames and the widgets that will be used in the user interface
    button_frame = customtkinter.CTkFrame(master=window)
    button_frame.pack(fill="x")
    frame1 = customtkinter.CTkFrame(master=window)
    frame2 = customtkinter.CTkFrame(master=window)

    button1 = customtkinter.CTkButton(button_frame, text="Search", command=lambda: switch_frame(frame1))  #this is the button that switches to the "search" frame
    button1.pack(side="left", padx=10, pady=10)

    button2 = customtkinter.CTkButton(button_frame, text="Inverted index", command=lambda: switch_frame(frame2))  #this is the button that switches to the "inverted index" frame
    button2.pack(side="left", padx=0, pady=10)

    frame1.columnconfigure(0, weight=1)
    frame1.columnconfigure(1, weight=0)
    frame1.columnconfigure(2, weight=0)
    frame1.columnconfigure(3, weight=0)
    frame1.rowconfigure(1, weight=1)

    frame2.columnconfigure(0, weight=1)
    frame2.columnconfigure(1, weight=0)
    frame2.columnconfigure(2, weight=0)
    frame2.columnconfigure(3, weight=0)
    frame2.rowconfigure(1, weight=1)

    query_entry = customtkinter.CTkEntry(master=frame1)  #this is the textbox where the user will enter the search query
    query_entry.grid(row=0, column=0, padx=10, pady=12, sticky="ew")

    sort_by_dropdown = customtkinter.CTkComboBox(master=frame1, values=["Tf-Idf", "Boolean retrieval", "Okapi BM25"]) #this is the dropdown where the user will select the sorting method
    sort_by_dropdown.grid(row=0, column=2, padx=10, pady=12, sticky="w")

    search = customtkinter.CTkButton(master=frame1, text="Search", width=20, command=search_and_update)  #this is the button that will search for the query
    search.grid(row=0, column=3, padx=10, pady=12, sticky="e")

    results_text = customtkinter.CTkTextbox(master=frame1, width=400, height=400)  #this is the textbox where the results will be displayed
    results_text.grid(row=1, column=0, columnspan=4, sticky="nsew")
    results_text.configure(state='disabled')

    boolean_search_entry = customtkinter.CTkEntry(master=frame2)  #this is the textbox where the user will enter the boolean search query
    boolean_search_entry.grid(row=0, column=0, padx=10, pady=12, sticky="ew")

    boolean_s = customtkinter.CTkButton(master=frame2, text="Boolean Search", width=20, command=boolean_search) #this is the button that will perform the boolean search
    boolean_s.grid(row=0, column=3, padx=10, pady=12, sticky="e")

    inverted_index_widget = customtkinter.CTkTextbox(master=frame2, width=400, height=400)  #this is the textbox where the inverted index will be displayed
    inverted_index_widget.grid(row=1, column=0, columnspan=4, sticky="nsew")
    inverted_index_widget.configure(state='disabled')

    switch_frame(frame1)
    window.mainloop()


def main():
    create_ui()


if __name__ == "__main__":
    main()
