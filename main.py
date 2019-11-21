import math
import sys

from nltk import word_tokenize


class CategoryInfo:
    def __init__(self, name, log_prior, word_prob_dictionary):
        self.name = name
        self.log_prior = log_prior
        self.word_prob_dictionary = word_prob_dictionary


# write information regarding each incorrectly tagged sentence to output file
def write_output(output_file, data):
    f = open(output_file, "w")

    for each in data:

        # output the sentence
        f.write(each.sentence + "\n")

        # output pos tag for each word in the sentence
        for word_tag_tuple in each.pos_tags:
            f.write(word_tag_tuple[0] + ' ' + word_tag_tuple[1] + "\n")

        # output all the incorrectly tagged entities
        for entity in each.incorrectly_tagged_entities:
            f.write(entity + "\n")

        # output two blank lines
        f.write("\n\n")

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

    return category_info_list, vocabulary


def test(category_info_list, vocabulary, test_doc):
    max_val = - sys.maxsize - 1

    max_category = None

    for x in category_info_list:

        sum_c = x.log_prior

        tokens = word_tokenize(test_doc)

        for token in tokens:

            if token in vocabulary:
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

        category_list = []
        for x in train_set:
            category_list.append(x[0])

        category_set = set(category_list)

        category_info_list, vocabulary = train(train_set, category_set)

        for x in test_set:
            best_category = test(category_info_list, vocabulary, x[1])
            print(x[0] + ' ' + best_category)
        break


def main():
    if len(sys.argv) < 2:
        print("Invalid number of arguments.")
        return
    else:
        train_file_path = sys.argv[1]

    with open(train_file_path) as f:
        content = f.read()
        processed_sample_list = preprocess_data(content)

    fold_list = fold_data(processed_sample_list)

    cross_validation(fold_list)


if __name__ == "__main__":
    main()
