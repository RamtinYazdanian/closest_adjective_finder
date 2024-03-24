from nltk.corpus import wordnet as wn
import numpy as np
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--new', type=str, required=True)
    parser.add_argument('--number', type=int, default=100)
    args = parser.parse_args()
    original_word = args.original
    new_word = args.new
    n = args.number
    print(f'Original: {original_word}')
    print(f'New: {new_word}')

    model = SentenceTransformer("all-MiniLM-L6-v2")
    w1 = model.encode(original_word)
    w2 = model.encode(new_word)

    all_adj = list(
        set(
            [word for synset in wn.all_synsets('a') for word in synset.lemma_names()]
            + [word for synset in wn.all_synsets('s') for word in synset.lemma_names()]
        )
    )
    print(f'Number of all English adjectives: {len(all_adj)}')
    all_adj_embeddings = model.encode(all_adj)

    ind_1 = np.argsort(util.cos_sim(w1, all_adj_embeddings)).flatten().tolist()[-n:]
    ind_2 = np.argsort(util.cos_sim(w2, all_adj_embeddings)).flatten().tolist()[-n:]

    closest_1 = [all_adj[i] for i in ind_1]
    closest_2 = [all_adj[i] for i in ind_2]

    print('Closest adjectives to original word:')
    print(closest_1)
    print()

    print('Closest adjectives to new word:')
    print(closest_2)
    print()

    print('Adjectives close to the original word but not the new one:')
    print(set(closest_1) - set(closest_2))
    print()

    print('Adjectives close to the new word but not the original:')
    print(set(closest_2) - set(closest_1))
    print()


if __name__ == '__main__':
    main()
