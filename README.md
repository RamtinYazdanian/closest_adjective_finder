# Closest adjective set comparer

Using this Python script, you can compare, for two English words, the sets of the closest adjectives to each of them and the difference between these two sets. For example, you could compare the closest adjectives to "homeless" and "unhoused".

This script uses the WordNet database from the Python `nltk` package to get a list of English adjectives, and uses the pretrained Bert SentenceTransformer models for word embeddings.

## Installation

First install Python 3.11. On Mac:
```
brew install python@3.11
```

Then preferably create a new virtual environment and activate it:
```
python3.11 -m venv "py311venv"
source ./py311venv/bin/activate
```

Then install the required python packages:
```
pip install -r requirements.txt
```

## Usage

```
python find_unrelated_words.py --original homeless --new unhoused
```

In order to change the number of top adjectives, provide the `--number` argument, which is 100 by default:

```
python find_unrelated_words.py --original addict --new "Breaking Bad reenactor" --number 50
```

### Example results
```
Original: homeless
New: unhoused
Number of all English adjectives: 21538
Closest adjectives to original word:
['subterranean', 'huddled', 'nonsocial', 'Utopian', 'utopian', 'miserable', 'unsized', 'clothesless', 'desolate', 'alone', 'unpledged', 'shining', 'barefoot', 'unsocial', 'underground', 'lidless', 'haunted', 'migrant', 'shoeless', 'friendless', 'begotten', 'bodyless', 'depressed', 'spendthrift', 'suburbanised', 'unavenged', 'relaxing', 'Palestinian', 'unpaid', 'footless', 'pedestrian', 'runaway', 'secluded', 'apocalyptic', 'selfish', 'sober', 'rentable', 'abandoned', 'rootless', 'clean-living', 'troubled', 'sleeping', 'hopeless', 'bedless', 'drugless', 'drug-addicted', 'naked', 'hipless', 'home', 'jobless', 'vacant', 'comfortless', 'stranded', 'feral', 'starving', 'landless', 'rural', 'unaged', 'derelict', 'painless', 'boneless', 'non-living', 'stray', 'sheltered', 'nomadic', 'residential', 'free-living', 'roadless', 'poor', 'urban', 'moneyless', 'dilapidated', 'childless', 'inhabited', 'barren', 'mayoral', 'humanitarian', 'sleepless', 'nonprofit', 'adrift', 'charitable', 'lone', 'living', 'lonely', 'downtrodden', 'roofless', 'affluent', 'penitentiary', 'deserted', 'unemployed', 'volunteer', 'disadvantaged', 'solitary', 'needy', 'poverty-stricken', 'destitute', 'uninhabited', 'impoverished', 'untrodden', 'homeless']

Closest adjectives to new word:
['unthawed', 'unmutilated', 'unreverberant', 'unmanful', 'unrifled', 'unversed', 'unannealed', 'unfrozen', 'unhatched', 'unclouded', 'untied', 'unbloody', 'unpatented', 'unlubricated', 'unshapely', 'unholy', 'unsubdued', 'unifilar', 'unfettered', 'unshorn', 'unabridged', 'unnoted', 'unscripted', 'unrevealed', 'unsound', 'ungroomed', 'ungarmented', 'unsold', 'unrewarding', 'untalented', 'unshrinking', 'unsalted', 'uncaring', 'ungentle', 'unsent', 'unpitying', 'ungentlemanly', 'unpermed', 'uncultured', 'unfed', 'unabated', 'uncrowded', 'unpersuaded', 'unrhymed', 'untraversed', 'unhinged', 'unladylike', 'unanimated', 'unsubmissive', 'unworldly', 'unrhetorical', 'ungusseted', 'unreduced', 'unsleeping', 'untrimmed', 'unready', 'unasked', 'unplanned', 'unfathomed', 'uncombed', 'unlaced', 'uncoerced', 'unstaged', 'unrenewed', 'unwomanly', 'untried', 'uncarpeted', 'unpierced', 'unshoed', 'unspaced', 'unmanly', 'unsugared', 'unwooded', 'unshackled', 'untwisted', 'unroofed', 'unhuman', 'unchained', 'unbeneficed', 'uncared-for', 'unironed', 'untutored', 'uncreased', 'unbodied', 'unsullied', 'unforced', 'unconsidered', 'unvaned', 'unworried', 'unmotorised', 'unfretted', 'unhampered', 'unfueled', 'untidy', 'unweaned', 'ungeared', 'unshod', 'untrod', 'unpeopled', 'unmated']

Adjectives close to the original word but not the new one:
{'humanitarian', 'subterranean', 'depressed', 'destitute', 'pedestrian', 'Utopian', 'jobless', 'starving', 'unpledged', 'migrant', 'penitentiary', 'rural', 'clothesless', 'rootless', 'affluent', 'unavenged', 'shoeless', 'runaway', 'sleeping', 'sleepless', 'needy', 'nonsocial', 'naked', 'abandoned', 'sheltered', 'comfortless', 'hopeless', 'feral', 'shining', 'residential', 'derelict', 'bedless', 'disadvantaged', 'utopian', 'underground', 'Palestinian', 'hipless', 'non-living', 'dilapidated', 'nonprofit', 'haunted', 'sober', 'boneless', 'urban', 'lone', 'unsized', 'volunteer', 'inhabited', 'painless', 'barren', 'alone', 'solitary', 'begotten', 'unsocial', 'vacant', 'drugless', 'friendless', 'barefoot', 'uninhabited', 'mayoral', 'adrift', 'spendthrift', 'secluded', 'troubled', 'moneyless', 'unpaid', 'childless', 'downtrodden', 'rentable', 'free-living', 'landless', 'drug-addicted', 'poor', 'footless', 'lidless', 'unaged', 'suburbanised', 'relaxing', 'living', 'huddled', 'charitable', 'unemployed', 'impoverished', 'roofless', 'home', 'stray', 'desolate', 'deserted', 'lonely', 'untrodden', 'apocalyptic', 'roadless', 'poverty-stricken', 'stranded', 'clean-living', 'selfish', 'miserable', 'homeless', 'nomadic', 'bodyless'}

Adjectives close to the new word but not the original:
{'unwooded', 'unreduced', 'ungentlemanly', 'unworldly', 'unclouded', 'unreverberant', 'unvaned', 'unmotorised', 'untried', 'unhinged', 'unsent', 'untalented', 'unbeneficed', 'unpitying', 'unrenewed', 'unready', 'unpermed', 'untrimmed', 'unrewarding', 'unsold', 'unshoed', 'unmutilated', 'unfueled', 'unsound', 'unroofed', 'unshrinking', 'unasked', 'unsubdued', 'uncombed', 'unsubmissive', 'unlaced', 'unsullied', 'uncarpeted', 'uncreased', 'unshackled', 'unhampered', 'unpersuaded', 'unmanly', 'unrhymed', 'unthawed', 'unshorn', 'unabridged', 'unlubricated', 'unscripted', 'untidy', 'untrod', 'unannealed', 'unstaged', 'unifilar', 'unforced', 'untwisted', 'unironed', 'unfettered', 'unholy', 'unplanned', 'untraversed', 'unchained', 'unspaced', 'unconsidered', 'unweaned', 'uncoerced', 'unladylike', 'unbodied', 'unworried', 'unabated', 'ungarmented', 'ungroomed', 'unhuman', 'unnoted', 'ungeared', 'ungusseted', 'unrhetorical', 'unsalted', 'unwomanly', 'unfed', 'unpatented', 'ungentle', 'uncaring', 'unmated', 'untied', 'unrevealed', 'unshapely', 'unrifled', 'untutored', 'unanimated', 'unversed', 'unmanful', 'unfretted', 'unhatched', 'unpierced', 'uncrowded', 'unsleeping', 'unsugared', 'unbloody', 'uncultured', 'unfathomed', 'unfrozen', 'unshod', 'uncared-for', 'unpeopled'}
```
