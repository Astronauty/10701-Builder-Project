from data_loader import *
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""path = Path(f'data/Conan Data/en_lang_90.pickle')
with open(path, 'rb') as handle:
    enLang = pickle.load(handle)



max_mult_of_gap_size = 10
gap_size = 1
no_of_words = 0
words_to_keep = []
number_in_range_list = numpy.zeros(max_mult_of_gap_size + 1)
ghundred = 0
for word in enLang.word2count.keys():
    occurences = enLang.word2count[word]
    print(word)
    interval_no = int(numpy.floor(occurences/gap_size))
    if interval_no > max_mult_of_gap_size:
        ghundred +=1
        interval_no = max_mult_of_gap_size
    number_in_range_list[interval_no] += 1


print(ghundred)
print(number_in_range_list)"""

number_in_range_list = [0, 1400192,   374580,   156875,    93645,   60308,   43306,   33139, 26211,   21258,  278854]

plt.style.use('_mpl-gallery')

x = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10 or more"]
y = number_in_range_list[1:]

# plot
fig = plt.figure(figsize = (10, 5))

plt.bar(x, y)
plt.xlabel("Number x of times a word occurs")
plt.ylabel("Number of words that occur x times")
# this locator puts ticks at regular intervals
#loc = ticker.MultipleLocator(base=10)
plt.savefig('distributionTrials.png',bbox_inches='tight')






