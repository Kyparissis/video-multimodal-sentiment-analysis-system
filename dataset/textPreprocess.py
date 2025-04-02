# This is a Python file that contains functions for preprocessing text data
# It is used mainly for preprocessing the text data in the dataset CMU-MOSEI (labels.csv)
# --------------------------------------------------------------------------------------------

## CONTRACTIONS
# The following code fixes any contractions in the sentence
# eg. "I'm" -> "I am"
# eg. "I've" -> "I have"
# eg. "I'd" -> "I would"
# eg. "I'll" -> "I will"
# eg. "I'd" -> "I would"
# Also fixes any known slangs in the sentence (since we work with spoken language data)
# eg. "wanna" -> "want to"
# eg. "gonna" -> "going to"
# eg. "gotta" -> "got to"
# ΙΝPUT: sentence (string)
# OUTPUT: sentence with contractions expanded (string)
#        eg. "I'm going to the store." -> "I am going to the store."
import contractions
def expandContractions(sentence):
    return contractions.fix(sentence, 
                            slang=True)

# NORMALIZE APOSTROPHES
# This function normalizes apostrophes in the sentence
# Apostrophes are used to indicate possession e.g., "dog's"
# This function removes the apostrophe and the 's' if it is followed by an 's' (e.g., "dog's" -> "dog")
# It works for both regular apostrophes (') and curly apostrophes (’) (found in the dataset)
# INPUT: sentence (string)
#        eg. "This is my dog’s toy."
# OUTPUT: sentence with normalized apostrophes (string)
#         eg. "This is my dog toy."
import re
def normalizeApostrophes(sentence):
    # Define a regex pattern to match apostrophes before or after 's'
    # This pattern will:
    # - Remove 's' if it follows an apostrophe
    # - Remove the apostrophe itself if it is followed by 's'
    # How the regex pattern works:
    # - (\w+)’s\b matches a word character followed by '’s' at the end of a word
    # - (\w+)’\b matches a word character followed by '’' at the end of a word
    # - re.IGNORECASE makes the pattern case-insensitive
    pattern = re.compile(r"(\w+)[’']s\b|(\w+)[’']\b", re.IGNORECASE)
    
    def replace_apostrophe(match):
        # Extract the matched groups
        word_with_apostrophe = match.group(1) or match.group(2)
        return word_with_apostrophe

    # Replace occurrences using the regex pattern and the replacement function
    sentence = pattern.sub(replace_apostrophe, sentence)
    
    return sentence
    
# REMOVE 'CRAP'/FILLER WORDS / WORD FILTERING
# Filler words are words such as: "um", "uh" and variations of them (like "uhh", "umm", "uhhh", "ummm", etc.)
# This function removes filler words from the sentence. However, they have to be surrounded by square brackets or parentheses
# in order to avoid removing words like yummy, hummus, mummy, etc.
# INPUT: sentence (string)
#        eg. "Uhh this hard work is umm not worth it."
# OUTPUT: sentence with filler words removed (string)
#         eg. "this hard work is not worth it."
import re
def removeFillerWords(sentence):
    # List of common filler words and their flexible patterns
    filler_patterns = [
        r'\b\[?h?u+h+\]?\b',       # matches "uh", "uhh", "uhhh", etc.
        r'\b\(?h?u+h+\)?\b',
        r'\b\[?h?u+m+\]?\b',       # matches "um", "umm", "ummm", etc.
        r'\b\(?h?u+m+\)?\b',
        r'\b\[?h+m+\]?\b',       # matches "hmm", "hmmm", etc.
        r'\b\(?h+m+\)?\b',
    ]

    # Create a regex pattern to match the filler words
    # We use:
    # - re.IGNORECASE to make the pattern case-insensitive (accept both uhh and UHH and Uhh, etc.)
    # - re.compile() to compile the pattern into a regex object
    # - `|`.join(filler_patterns) to join the filler patterns with the OR operator. For example if the list contains
    # "uh" and "um", the joined pattern will be "uh|um
    # - r'(' + ... + r')' to group the joined pattern so that it is treated as a single pattern
    # the r is used to make the string a raw string, which is useful for regex patterns
    # - \s* to match any whitespace before the filler word. For example, it will match "uh" in " uh" or "  uh"
    # - ,? to match any comma after the filler word. For example, it will match "uh," in "uh" or "uh" in "uh"
    filler_pattern = re.compile(r'(' + '|'.join(filler_patterns) + r')\s*,?', 
                                re.IGNORECASE)

    # Use the sub() method to replace filler words with an empty string
    sentence = filler_pattern.sub('', sentence)

    # Remove any extra whitespace left after removing filler words
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence

# REMOVE URLS
# This function removes URLs from the sentence
# Such as: http://www.google.com, www.google.com, google.com, http://google.com, etc.
# Also works with URLs written in textual formats (e.g., "www dot example dot com")
# WARNING! It replaces it with a space character
# INPUT: sentence (string)
#        eg. "I don't know what to do. http://www.google.com is not working."
# OUTPUT: sentence with URLs removed (string)
#         eg. "I don't know what to do. is not working."
import re
def removeURL(sentence):
    # Define a regex pattern to match standard URLs
    url_pattern = re.compile(
        r'(https?://\S+|www\.\S+|'
        r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b/\S*)', re.IGNORECASE
    )
    
    # Define a regex pattern to match URLs written in textual formats (e.g., "www dot example dot com")
    textual_url_pattern = re.compile(
        r'\b(?:www|www\s+dot\s+)[\w\s]+(?:dot\s+[\w\s]+)+', re.IGNORECASE
    )
    
    # Remove standard URLs
    sentence = url_pattern.sub(' ', sentence)
    
    # Remove textual URLs
    sentence = textual_url_pattern.sub(' ', sentence)
    
    # Optionally, clean up any extra spaces that might have been left behind
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    return sentence

# REMOVE PARALINGUISTIC NOTATIONS
# Paralinguistic notations are text within brackets or parentheses that are used to represent non-verbal sounds or actions
# Such expressions are often used in spoken language data to represent laughter, pauses, etc.
# Examples are [scoffs], [laughs], [coughs], [sighs], [clears throat], etc.
# Works for square brackets, parentheses and curly brackets
# This function also works for elimination speaker tags like [Speaker 1], [Speaker 2], etc.
# INPUT: text (string)
#        eg. "I don't know what to do [sighs] I'm so confused."
# OUTPUT: text with paralinguistic notations removed (string)
#         eg. "I don't know what to do I'm so confused."
def removeParalinguisticNotations(sentence):
    # Define the regex pattern to match text within brackets or parentheses
    # How the pattern works:
    # - \[.*?\] matches any text within square brackets
    # - \(.*?\) matches any text within parentheses
    # - \{.*?\} matches any text within curly brackets
    # - | is the OR operator, so the pattern will match any of the three types of brackets
    pattern = r'\[(.*?)\]|\((.*?)\)|\{(.*?)\}'

    # Use re.sub() to replace the matched patterns with an empty string
    sentence = re.sub(pattern, '', sentence)

    # Remove any extra whitespace left after removing the notations
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence

## LOWERCASING
# This function converts the sentence to lowercase
# INPUT: sentence (string)
#        eg. "Hello, World!"
# OUTPUT: sentence in lowercase (string)
#         eg. "hello, world!"
def toLowercase(sentence):
    return sentence.lower()

# REMOVE PUNCTUATION
# This function removes punctuation from the sentence
# Such as: . , ! ? ; : ' " ( ) [ ] { } < > / \ | _ - + = * & % $ # @ ~ ` ^
# WARNING! It replaces it with a space character
# INPUT: sentence (string)
#        eg. "Hello, World!"
# OUTPUT: sentence with punctuation removed (string)
#         eg. "Hello World "
import string
def removePunctuation(sentence):
    PUNCT_TO_REMOVE = string.punctuation    # This variable holds all the punctuation characters 
                                            # !"#$%&'()*+,-./:;<=>?@[\]^_{|}~
                                            
    EXTRA_PUNCT_TO_REMOVE = "”" + "“" + "’" + "‘" + "—" + "–" + "®"
    PUNCT_TO_REMOVE += EXTRA_PUNCT_TO_REMOVE  # Additional punctuation found in the dataset

    # Create a regex pattern to match any of the punctuation characters
    pattern = re.compile(f'[{re.escape(PUNCT_TO_REMOVE)}]')

    # Replace each punctuation character with a space, but leave the words intact
    sentence = pattern.sub(' ', sentence)

    # Optionally, you can also strip excess spaces from the result
    sentence = ' '.join(sentence.split())

    return sentence

# STEMMING
# Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
# For example, if there are two words in the corpus walks and walking, then stemming will stem the suffix to make them walk. 
# But say in another example, we have two words console and consoling, the stemmer will remove the suffix and make them 
# consol which is not a proper english word.
# There are several type of stemming algorithms available and one of the famous one is porter stemmer which is widely used.
# SOURCE: https://www.datacamp.com/tutorial/stemming-lemmatization-python 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Punkt is not found. Downloading now...")
    nltk.download('punkt')
# INPUT: sentence (string)
# OUTPUT: stemmed sentence (string)
def stemming(sentence):
    stemmer = PorterStemmer()   # Stemmer object to stem the words

    # This function tokenizes the sentence into words
    # eg. "The quick fox is running above the wall and he found a cat." 
    # -> ["The", "quick", "fox", "is", "running", "above", "the", "wall", "and", "he", "found", "a", "cat", "."]
    words = word_tokenize(sentence)

    # Stem each word in the sentence
    stemmed_words = [stemmer.stem(word) for word in words]

    # Return the joined stemmed words
    return ' '.join(stemmed_words)

# LEMMATIZATION
# Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item
# eg. "running" -> "run"
# eg. "better" -> "good"
# Lemmatization is more accurate than stemming because it uses a vocabulary and morphological analysis of words.
# Keeps in mind the meaning of the word in the sentence.
# However, it is computationally expensive and slower than stemming.
# SOURCE: https://www.datacamp.com/tutorial/stemming-lemmatization-python
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    print("Wordnet is not found. Downloading now...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
# INPUT: sentence (string)
# OUTPUT: lemmatized sentence (string)
def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()   # Stemmer object to stem the words

    # This function tokenizes the sentence into words
    # eg. "The quick fox is running above the wall and he found a cat." 
    words = word_tokenize(sentence)

    # Lemmatize each word in the sentence
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Return the joined lemmatized words
    return ' '.join(lemmatized_words)

# NUMBERS TO WORDS
# This function converts numbers to words. It works for both integers and floats.
# It works even if the number is written with commas (e.g., 32,000) or if it is followed by a decimal point (e.g., 32,000.5)
# INPUT: sentence (string)
#        eg. "The price is $32,001.5 and it is not worth it."
# OUTPUT: sentence with numbers converted to words (string)
#         eg. "The price is thirty-two thousand one point five and it is not worth it."
import inflect
import re

def number2words(sentence, toRemove=False):
    p = inflect.engine()

    # Split the sentence into words
    words = sentence.split()

    # Initialize a list to hold the converted words
    converted_words = []

    # Regular expression to match numbers with optional punctuation
    pattern = re.compile(r'(\d+)([^\d\s]?)')

    # Iterate through each word in the sentence
    for word in words:
        match = pattern.match(word)

        if match:
            num_part = match.group(1)
            trailing_char = match.group(2)

            # Convert the numeric part to words
            try:
                num = int(num_part)
                converted_word = p.number_to_words(num)

                # Add the trailing punctuation, if any
                if trailing_char:
                    converted_word += trailing_char

                if toRemove:
                    converted_words.append('')
                else:
                    converted_words.append(converted_word)
            except ValueError:
                converted_words.append(word)
        else:
            converted_words.append(word)

    # Join the list of converted words back into a sentence
    return ' '.join(converted_words)

# ORDINALS TO WORDS
# This function converts ordinal numbers to words
# Ordinal numbers are numbers that denote a position in a sequence, such as first, second, third, etc.
# eg. "1st" -> "first"
# eg. "2nd" -> "second"
# eg. "3rd" -> "third"
# eg. "4th" -> "fourth"
# INPUT: sentence (string)
#       eg. "The 1st, 2nd, 3rd and 4th place winners are all here."
# OUTPUT: sentence with ordinal numbers converted to words (string)
#        eg. "The first, second, third and fourth place winners are all here."
import inflect
import re
def ordinals2words(sentence):
    p = inflect.engine()
    
    # Define a regex pattern to match ordinal numbers
    # How the pattern works:
    # - \b matches a word boundary, so it will match the start and end of a word
    # - (\d+) matches one or more digits and captures them in a group
    # - (st|nd|rd|th) matches one of the ordinal suffixes and captures it in a group
    # - \b matches another word boundary
    pattern = r'\b(\d+)(st|nd|rd|th)\b'
    
    # Function to convert matched ordinal numbers
    def replace_ordinal(match):
        number = int(match.group(1))                    # Extract the number part
        return p.ordinal(p.number_to_words(number))     # Convert to words with ordinal suffix
    
    # Replace ordinal numbers in the text with their word equivalents
    sentence = re.sub(pattern, replace_ordinal, sentence)
    
    return sentence

# RANGES
# This function converts numeric ranges to words
# For example, "Price range: £150-2,000.5" -> "Price range: one hundred fifty to two thousand point five"
# INPUT: range_text (string)
# OUTPUT: range_text with numeric ranges converted to words (string)
import inflect
import re
def range2words(range_text, toRemove=False):
    p = inflect.engine()

    # Define a regex pattern to match numeric ranges, handling numbers with commas and optional decimal points
    # How the pattern works analytically part by part:
    # - (\d{1,3}) matches one to three digits (e.g., 1, 10, 100, 1000)
    # - (,\d{3})* matches zero or more occurrences of a comma followed by three digits (e.g., 1,000, 10,000, 100,000)
    # - (\.\d+)? matches an optional decimal point followed by one or more digits (e.g., .5, .25, .123)
    # - - matches a hyphen to separate the start and end of the range
    # - The same pattern is repeated to match the end of the range   
    pattern = r'(\d{1,3}(,\d{3})*(\.\d+)?)-(\d{1,3}(,\d{3})*(\.\d+)?)'

    # Function to convert the matched range to words
    def convert_range(match):
        # Extract the start and end of the range, removing commas
        start = match.group(1).replace(',', '')
        end = match.group(4).replace(',', '')

        # Convert the numbers to words using the number2words function
        start_word = number2words(start)
        end_word = number2words(end)

        if toRemove:
            return ''
        else:
            # Return the formatted text without the currency
            return f"{start_word} to {end_word}"

    # Use regex to replace the range with its word equivalent
    result = re.sub(pattern, convert_range, range_text)

    return result

# REMOVE CURRENCY SYMBOLS
# This function removes currency symbols from the sentence
# Such as: $ € £ ¥ ₹ ¢ ₩ ₽ ฿ ₦ ₫ ₱ ₲ ₴ ₮
# It works even if the symbol is followed by a number (e.g., $32,000) or if it is after the number (e.g., 32,000$)
# INPUT: sentence (string)
#        eg. "The price is $32,000 and it is not worth it."
# OUTPUT: sentence with currency symbols removed (string)
#         eg. "The price is 32,000 and it is not worth it."
def removeCurrencySymbols(sentence):
    # Define a regular expression pattern to match any currency symbol
    pattern = r'[\$\€\£\¥\₹\¢\₩\₽\฿\₦\₫\₱\₲\₴\₮]'
    
    # Use re.sub to replace currency symbols with an empty string
    sentence = re.sub(pattern, '', sentence)
    
    return sentence

# Convert percentages and numbering/rankingto words
# This function converts percentages in the text to words
# eg. "92%" -> "92 percent"
# eg. "I'm the #1 and I'm 92% sure about it." -> " I'm the number 1 and I'm 92 percent sure about it.
# INPUT: sentence (string)
# OUTPUT: sentence with percentages and rankings converted to words (string)
def replacePercentagesAndNumbers(sentence, toRemove=False):
    # Regex pattern to match percentages like "92%", "100%", "7%"
    percentage_pattern = r'(\d+)%'
    
    # Regex pattern to match numbers like "#1", "#100"
    number_pattern = r'#(\d+)'
    
    if toRemove:
        # Replace percentages with "X percent"
        sentence = re.sub(percentage_pattern, '', sentence)
        
        # Replace "#X" with "number X"
        sentence = re.sub(number_pattern, '', sentence)
    else:
        # Replace percentages with "X percent"
        sentence = re.sub(percentage_pattern, r'\1 percent', sentence)
        
        # Replace "#X" with "number X"
        sentence = re.sub(number_pattern, r'number \1', sentence)
    
    return sentence

# Remove words with specific substrings
# This function removes words from a sentence that contain any of the specified substrings provided by me manually
# eg. if the substrings are ["&amp", ...] (List is created by observing the dataset), then 
# "This is an &amp; example &ampsentence with some test words." ->  "This is an example with test words."
# INPUT: sentence (string)
# OUTPUT: sentence with words containing the substrings removed (string)
import re
def removeWordsWithSubstrings(sentence):
    # Split the sentence into words
    words = sentence.split()

    substrings = ["&amp", 
                  ]
    
    # Filter words that do not contain any of the substrings
    filtered_words = [word for word in words if not any(sub in word for sub in substrings)]
    
    # Join the filtered words back into a sentence
    return ' '.join(filtered_words)

# REMOVE STOPWORDS
# Stopwords are common words that do not contribute much to the meaning of a sentence
# For example, "the", "is", "at", "which", "on", "for", "this", etc.
# eg. "This is a test sentence." -> "This test sentence."
import nltk
from nltk.corpus import stopwords
# Check if stopwords data is already downloaded, if not download it
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')
# INPUT: sentence (string)
# OUTPUT: sentence with stopwords removed (string)
def removeStopwords(sentence):
    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Split the sentence into words
    words = sentence.split()
    
    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a sentence
    return ' '.join(filtered_words)

# SPELL CHECKING
# Pure Python Spell Checking based on Peter Norvig's blog post on setting up a simple spell checking algorithm.
# It uses a Levenshtein Distance algorithm to find permutations within an edit distance of 2 from the original word. 
# It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a 
# word frequency list. Those words that are found more often in the frequency list are more likely the correct results.
# SOURCE: https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing#Spelling-Correction
#         https://github.com/barrust/pyspellchecker
# INPUT: sentence (string)
#        eg. "This   is  a sentence"
# OUTPUT: correct spacing sentence (string)
#        eg. "This is a sentence"
from spellchecker import SpellChecker
def spellCorrect(sentence):
    spell = SpellChecker()        # Initialize the spell checker
    words = sentence.split()      # Split the sentence into individual words
    corrected_sentence = []

    for word in words:
        corrected_word = spell.correction(word)     # Get the corrected word
        corrected_sentence.append(corrected_word)   # Add the corrected word to the list

    return ' '.join(corrected_sentence)  

# Remove non-alphanumeric words
# This function removes from a sentence any words that contain anything other than a-z, A-Z, or 0-9
# eg. "this is so lovelÿ" -> "this is so"
# INPUT: text (string)
# OUTPUT: cleaned_text (string)
import re
# TODO:
# FIX THIS FUNCTION SO IT DOESNT REMOVE THE WHOLE WORD. 
# MAKE IT REMOVE THE CHARACTER AND THEN USE THE spellCorrect FUNCTION TO CORRECT THE WORD
# eg. "this is so lovelÿ" -> "this is so lovely"
# BUT SHOULD STILL WORK FOR SENTENCE LIKE:
# “— +,ù®0 0 h p ˜ &nbsp; ¨ ° ¸ À È Ð ä Mount Holyoke College ' 6 Is financial aid available for international students Title þÿÿÿ þÿÿÿ þÿÿÿ ! " # þÿÿÿ% &amp; ' ( ) * + þÿÿÿýÿÿÿ.
def removeNonAlphanumericWords(sentence):
    # Use regex to remove words that contain anything other than a-z, A-Z, or 0-9
    pattern = r'^[a-zA-Z0-9]+$'
    
    # Split the sentence into words
    words = sentence.split()

    # Filter words that match the pattern
    filtered_words = [word for word in words if re.match(pattern, word)]

    # Join the filtered words back into a sentence (here we join with a space character)
    sentence = ' '.join(filtered_words)

    return sentence

# FIX SPACING BETWEEN WORDS
# This function takes as input a sentence with wrong spacing and returns the same sentence with the correct
# spacing, putting only 1 space character between each word of the sentence
# INPUT:
# sentence (string): sentence with wrong spacing
#       eg. "This   is  a sentence"
# OUTPUT:
# sentence (string): sentence with correct spacing
#       eg. "This is a sentence"
def fixSpacing(sentence):
    # Split the sentence by whitespace and join it back with a single space
    sentence = ' '.join(sentence.split())

    return sentence

# --------------------------------------------------------------------------------------------
# MAIN FUNCTION
# This main function preprocesses the sentence in the whole dataset
# The function gives the user the ability to choose which preprocessing steps to apply
# INPUT: sentence (string)
#        expand_contractions (bool):                whether to expand contractions or not
#        replace
# OUTPUT: preprocessed sentence (string)
def sentencePreprocess(sentence, 
                       removeNumbers=True,
                       expand_contractions=True,
                       replace_percentages_and_numbers=True,
                       remove_URL=True,
                       normalize_apotrophes=True,
                       remove_currency_symbols=True,
                       range_to_words=True,
                       numbers_to_words=True,
                       ordinals_to_words=True,
                       remove_filler_words=True,
                       remove_paralinguistic_notations=True,
                       to_lowercase=True,
                       remove_stopwords=False,
                       to_stem=False,               # Why lemm instead of stem?
                       to_lemmatize=True,           # Better accuracies
                       to_filter=True,
                       remove_punctuation=True,
                       remove_non_alphanumeric=True,
                       spell_correct=False,
                       fix_spacing=True):
    # The order in which the preprocessing steps are applied is important
    # Don't change the call order of the functions below since it might affect the final result!!!

    # Convert the sentence to lowercase
    if to_lowercase:
        sentence = toLowercase(sentence)

    # Remove stopwords
    if remove_stopwords:
        sentence = removeStopwords(sentence)
    
    # Expand contractions
    if expand_contractions:
        sentence = expandContractions(sentence)

    # Convert percentages and numbering/ranking to words
    if replace_percentages_and_numbers:
        sentence = replacePercentagesAndNumbers(sentence, toRemove=removeNumbers)

    # Remove URLs
    if remove_URL:
        sentence = removeURL(sentence)

    # Normalize apostrophes
    if normalize_apotrophes:
        sentence = normalizeApostrophes(sentence)

    # Remove words with specific substrings
    if to_filter:
        sentence = removeWordsWithSubstrings(sentence)

    # Remove currency symbols
    if remove_currency_symbols:
        sentence = removeCurrencySymbols(sentence)

    # Convert ranges to words
    if range_to_words:
        sentence = range2words(sentence, toRemove=removeNumbers)
    
    # Convert numbers to words
    if numbers_to_words:
        sentence = number2words(sentence, toRemove=removeNumbers)

    # Convert ordinals to words
    if ordinals_to_words:
        sentence = ordinals2words(sentence)
    
    # Remove filler words
    if remove_filler_words:
        sentence = removeFillerWords(sentence)
    
    # Remove paralinguistic notations
    if remove_paralinguistic_notations:
        sentence = removeParalinguisticNotations(sentence)

    # Lemmatize the sentence
    if to_lemmatize:
        sentence = lemmatization(sentence)
        to_stem = False
    # OR
    # Stem the sentence
    if to_stem:
        sentence = stemming(sentence)
        to_lemmatize = False

    # Remove punctuation
    if remove_punctuation:
        sentence = removePunctuation(sentence)

    # Remove non-alphanumeric words
    if remove_non_alphanumeric:
        sentence = removeNonAlphanumericWords(sentence)

    # Spell Correction
    # DEFAULT = False (It tends to change names to false words eg. Penelope Cruz -> Envelope Crud)
    if spell_correct:
        sentence = spellCorrect(sentence)

    if fix_spacing:
        sentence = fixSpacing(sentence)
    
    return sentence

print(sentencePreprocess("FROM THERE IT DOES GET A BIT LIKE HUH NOT AS REAL AS WHAT HAS BEEN GOING ON"))