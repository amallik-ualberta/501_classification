import math
import random
import sys

from nltk import word_tokenize


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


def fold_data(data):
    random.shuffle(data)
    first_split_point = int(len(data) / 3)

    second_split_point = 2 * int(len(data) / 3)

    fold1 = data[:first_split_point]

    fold2 = data[first_split_point:second_split_point]

    fold3 = data[second_split_point:]

    fold_list = [fold1, fold2, fold3]

    return fold_list


def train(train_set, category_set):
    word_list = []
    for x in train_set:
        tokens = word_tokenize(x[1])
        word_list += tokens

    vocabulary = set(word_list)

    category_info_list = []
    for category in category_set:

        category_count = 0
        word_prob_dictionary = {}

        bigdoc = []

        for x in train_set:

            if x[0] == category:
                tokens = word_tokenize(x[1])
                bigdoc += tokens

                category_count += 1

        log_prior = math.log2(category_count / len(train_set))

        category_word_count = len(bigdoc)

        for word in vocabulary:
            word_count = bigdoc.count(word)

            log_likelihood = math.log2((word_count + 1) / (category_word_count + len(vocabulary)))

            word_prob_dictionary[word] = log_likelihood

        category_info = CategoryInfo(category, log_prior, word_prob_dictionary)
        category_info_list.append(category_info)
        model = Model(category_info_list, vocabulary)

    return model


def test(model, test_doc):
    max_val = - sys.maxsize - 1

    max_category = None

    for x in model.category_info_list:

        sum_c = x.log_prior

        tokens = word_tokenize(test_doc)

        for token in tokens:

            if token in model.vocabulary:
                sum_c += x.word_prob_dictionary[token]

        if sum_c > max_val:
            max_val = sum_c

            max_category = x.name

    return max_category


def cross_validation(fold_list):
    for i in range(0, 3):

        test_set = fold_list[i]

        train_set = []

        for x in fold_list:

            if x != test_set:
                train_set += x

        accuracy, model, output_data_list = train_test(train_set, test_set)
        print("Train Accuracy: %s" % accuracy)


def train_test(train_set, test_set):
    category_list = []
    for x in train_set:
        category_list.append(x[0])

    category_set = set(category_list)

    model = train(train_set, category_set)

    correct_category_count = 0
    output_data_list = []
    for x in test_set:
        best_category = test(model, x[1])

        output_data = OutputData(x[0], best_category, x[1])
        output_data_list.append(output_data)

        if x[0] == best_category:
            correct_category_count += 1

    accuracy = correct_category_count / len(test_set) * 100

    return accuracy, model, output_data_list


def eval(model, eval_set):
    output_data_list = []
    for x in eval_set:
        best_category = test(model, x[1])
        output_data = OutputData('', best_category, x[1])
        output_data_list.append(output_data)

    return output_data_list


def main():
    if len(sys.argv) < 4:
        print("Invalid number of arguments.")
        return
    else:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]
        eval_file_path = sys.argv[3]

    with open(train_file_path) as f:
        train_content = f.read()
        processed_train_sample_list = preprocess_data(train_content)

    fold_list = fold_data(processed_train_sample_list)

    # cross validate on train data
    cross_validation(fold_list)

    # test on test data
    with open(test_file_path) as f:
        test_content = f.read()
        processed_test_sample_list = preprocess_data(test_content)

    test_accuracy, model, output_data_list = train_test(processed_train_sample_list, processed_test_sample_list)
    print("Test Accuracy: %s" % test_accuracy)
    write_output(get_file_name_excluding_extension(test_file_path), output_data_list)

    # evaluate on eval data
    with open(eval_file_path) as f:
        eval_content = f.read()
        processed_eval_sample_list = preprocess_data(eval_content)

    output_data_list = eval(model, processed_eval_sample_list)
    write_output(get_file_name_excluding_extension(eval_file_path), output_data_list)


if __name__ == "__main__":
    main()
