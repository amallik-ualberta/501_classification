import math
import sys

from nltk import word_tokenize
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

nltk_stopwords = set(stopwords.words('english'))


class CategoryInfo:
    def __init__(self, name, log_prior, word_prob_dictionary):
        self.name = name
        self.log_prior = log_prior
        self.word_prob_dictionary = word_prob_dictionary


class Model:
    def __init__(self, category_info_list, vocabulary):
        self.category_info_list = category_info_list
        self.vocabulary = vocabulary


class OutputData:
    def __init__(self, original_label, assigned_label, text):
        self.original_label = original_label
        self.assigned_label = assigned_label
        self.text = text


# write information regarding each sample to output file
def write_output(output_file, output_data):
    output_file = "output_" + output_file + ".csv"

    f = open(output_file, "w")

    f.write("original label,classifier-assigned label,text" + "\n")

    for each in output_data:
        f.write(each.original_label + "," + each.assigned_label + "," + each.text + "\n")

    f.close()


def remove_stop_words(word_list):
    filtered_word_list = []
    for word in word_list:
        if word not in nltk_stopwords:
            filtered_word_list.append(word)

    return filtered_word_list


# get file name excluding extension
def get_file_name_excluding_extension(file_path):
    # for windows machine only, replace '\' with '/' in file path
    file_path = file_path.replace("\\", '/')

    # if file path doesn't contain '/' then get the file name after splitting using '.'
    if file_path.find('/') == -1:
        filename_without_ext = file_path.rsplit('.', 1)[0]

    # if file path contains '/' then split the file name using '/' first and then using '.'
    else:
        filename = file_path.rsplit('/', 1)[1]
        filename_without_ext = filename.rsplit('.', 1)[0]

    return filename_without_ext


def preprocess_data(data):
    sample_list = data.split('\n')

    # removing the header
    del sample_list[0]

    processed_sample_list = []

    for sample in sample_list:

        if len(sample) > 0:
            category_text_tuple = tuple(sample.split(',', 1))

            processed_sample_list.append(category_text_tuple)

    return processed_sample_list


def train(train_set, category_set):
    model = None
    alpha = 0.1

    word_list = []
    for x in train_set:
        # tokens = word_tokenize(x[1])
        tokens = x[1].split()
        tokens = remove_stop_words(tokens)
        word_list += tokens

    vocabulary = set(word_list)

    category_info_list = []
    for category in category_set:

        category_count = 0
        word_prob_dictionary = {}

        bigdoc = []

        for x in train_set:

            if x[0] == category:
                # tokens = word_tokenize(x[1])
                tokens = x[1].split()
                tokens = remove_stop_words(tokens)
                bigdoc += tokens

                category_count += 1

        log_prior = math.log2(category_count / len(train_set))

        category_word_count = len(bigdoc)

        for word in vocabulary:
            word_count = bigdoc.count(word)

            log_likelihood = math.log2((word_count + alpha) / (category_word_count + len(vocabulary) * alpha))

            word_prob_dictionary[word] = log_likelihood

        category_info = CategoryInfo(category, log_prior, word_prob_dictionary)
        category_info_list.append(category_info)

        model = Model(category_info_list, vocabulary)

    return model


def classify(model, test_doc):
    max_val = - sys.maxsize - 1

    max_category = None

    for x in model.category_info_list:

        sum_c = x.log_prior

        # tokens = word_tokenize(test_doc)
        tokens = test_doc.split()
        tokens = remove_stop_words(tokens)

        for token in tokens:

            if token in model.vocabulary:
                sum_c += x.word_prob_dictionary[token]

        if sum_c > max_val:
            max_val = sum_c

            max_category = x.name

    return max_category


def fold_data(data):
    first_split_point = int(len(data) / 3)

    second_split_point = 2 * int(len(data) / 3)

    fold1 = data[:first_split_point]

    fold2 = data[first_split_point:second_split_point]

    fold3 = data[second_split_point:]

    fold_list = [fold1, fold2, fold3]

    fold_range_list = [(1, first_split_point),
                       (first_split_point + 1, second_split_point),
                       (second_split_point + 1, len(data))]

    return fold_list, fold_range_list


def cross_validation(train_data):
    fold_list, fold_range_list = fold_data(train_data)

    total_accuracy = 0
    for i in range(0, len(fold_list)):

        train_set = []
        dev_set = []

        print("######## Cross Validation %s ########" % (i + 1))
        print("Row Number 1 : Header")
        for index, fold_range in enumerate(fold_range_list):
            if index == i:
                dev_set = fold_list[index]
                print("Row Number %s to %s : Dev Set" % (fold_range[0] + 1, fold_range[1] + 1))

            else:
                train_set += fold_list[index]
                print("Row Number %s to %s : Training Set" % (fold_range[0] + 1, fold_range[1] + 1))

        model = get_trained_model(train_set)

        correct_category_count = 0

        for x in dev_set:
            best_category = classify(model, x[1])

            if x[0] == best_category:
                correct_category_count += 1

        accuracy = correct_category_count / len(dev_set) * 100
        print("Training Accuracy: %s\n" % accuracy)

        total_accuracy += accuracy

    print("Average Training Accuracy: %s\n" % (total_accuracy / len(fold_list)))


def get_trained_model(train_set):
    category_list = []
    for x in train_set:
        category_list.append(x[0])

    category_set = set(category_list)

    model = train(train_set, category_set)
    return model


# test the model on test data
def test(model, test_set):
    true_category_list = []
    predicted_category_list = []
    correct_category_count = 0
    output_data_list = []

    # classify test samples one by one
    for x in test_set:

        # classify the sample
        best_category = classify(model, x[1])

        # store output data to OutputData class
        output_data = OutputData(x[0], best_category, x[1])

        # append output data to the output data list
        output_data_list.append(output_data)

        # append to a list of true categories
        true_category_list.append(x[0])

        # append to a list of predicted categories
        predicted_category_list.append(best_category)

        # to count the instances where predicted category and true category are same
        if x[0] == best_category:
            correct_category_count += 1

    # calculate and print accuracy
    accuracy = correct_category_count / len(test_set) * 100
    print("Test Accuracy: %s" % accuracy)

    category_set = set(true_category_list)
    cm = confusion_matrix(true_category_list, predicted_category_list, list(category_set))
    print("Confusion Matrix:\n %s \n %s" % (list(category_set), cm))

    return output_data_list


# evaluate model on evaluation data
def evaluate(model, eval_set):
    # contains output data for all the evaluation samples
    output_data_list = []

    # classify samples one by one
    for x in eval_set:
        # classify the sample
        best_category = classify(model, x[1])

        # store output data in OuputData class
        output_data = OutputData('', best_category, x[1])

        # append output data to output data list
        output_data_list.append(output_data)

    return output_data_list


def main():
    # parse command line arguments (expecting: python3 main.py TRAIN_FILE_PATH TEST_FILE_PATH EVAL_FILE_PATH)
    if len(sys.argv) < 4:
        print("Invalid number of arguments. Use following command pattern:")
        print("python3 main.py TRAIN_FILE_PATH TEST_FILE_PATH EVAL_FILE_PATH")
        return
    else:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]
        eval_file_path = sys.argv[3]

    # read and preprocess training file
    with open(train_file_path) as f:
        train_content = f.read()

        # preprocess training data
        processed_train_sample_list = preprocess_data(train_content)

    # cross validate on training data
    # cross_validation(processed_train_sample_list)

    # train model on whole training data
    model = get_trained_model(processed_train_sample_list)

    # read and preprocess test file
    with open(test_file_path) as f:
        test_content = f.read()

        # preprocess test data
        processed_test_sample_list = preprocess_data(test_content)

    # test the model on test data
    output_data_list = test(model, processed_test_sample_list)

    # write test output to output file
    write_output(get_file_name_excluding_extension(test_file_path), output_data_list)

    # read and proprocess evaluation file
    with open(eval_file_path) as f:
        eval_content = f.read()

        # preprocess evaluation data
        processed_eval_sample_list = preprocess_data(eval_content)

    # evaluate the model on eval data
    output_data_list = evaluate(model, processed_eval_sample_list)

    # write evaluation output to output file
    write_output(get_file_name_excluding_extension(eval_file_path), output_data_list)


if __name__ == "__main__":
    main()
