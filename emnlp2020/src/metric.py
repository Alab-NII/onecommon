from collections import OrderedDict, defaultdict
import numpy as np
import pdb
import time
import re

from nltk import word_tokenize, bigrams

import data

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
sns.set(font_scale=1.15)


class TimeMetric(object):
    def __init__(self):
        self.t = 0
        self.n = 0

    def reset(self):
        self.last_t = time.time()

    def record(self, n=1):
        self.t += time.time() - self.last_t
        self.n += 1

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.t / self.n

    def show(self):
        return '%.3fs' % (1. * self.value())


class NumericMetric(object):
    def __init__(self):
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def record(self, k, n=1):
        self.k += k
        self.n += n

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.k / self.n


class PercentageMetric(NumericMetric):
    def show(self):
        return '%2.2f%%' % (100. * self.value())


class AverageMetric(NumericMetric):
    def show(self):
        return '%.2f' % (1. * self.value())


class MovingNumericMetric(object):
    def __init__(self, window=100):
        self.window = window
        self.a = np.zeros(window)
        self.n = 0

    def reset(self):
        pass

    def record(self, k):
        self.a[self.n % self.window] = k
        self.n += 1

    def value(self):
        s = np.sum(self.a)
        n = min(self.a.size, self.n + 1)
        return 1.0 * s / n


class MovingAverageMetric(MovingNumericMetric):
    def show(self):
        return '%.2f' % (1. * self.value())


class MovingPercentageMetric(MovingNumericMetric):
    def show(self):
        return '%2.2f%%' % (100. * self.value())


class TextMetric(object):
    def __init__(self, text):
        self.text = text
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def value(self):
        self.n = max(1, self.n)
        return 1. * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class NGramMetric(TextMetric):
    def __init__(self, text, ngram=-1):
        super(NGramMetric, self).__init__(text)
        self.ngram = ngram

    def record(self, sen):
        n = len(sen) if self.ngram == -1 else self.ngram
        for i in range(len(sen) - n + 1):
            self.n += 1
            target = ' '.join(sen[i:i + n])
            if self.text.find(target) != -1:
                self.k += 1


class UniquenessMetric(object):
    def __init__(self):
        self.seen = set()

    def reset(self):
        pass

    def record(self, sen):
        self.seen.add(' '.join(sen))

    def value(self):
        return len(self.seen)

    def show(self):
        return str(self.value())


class SimilarityMetric(object):
    def __init__(self):
        self.reset()
        self.k = 0
        self.n = 0

    def reset(self):
        self.history = []

    def record(self, sen):
        self.n += 1
        sen = ' '.join(sen)
        for h in self.history:
            if h == sen:
                self.k += 1
                break
        self.history.append(sen)

    def value(self):
        self.n = max(1, self.n)
        return 1. * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class WordFrequencyMetric(object):
    def __init__(self, keys=[]):
        self.keys = keys
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def record(self, sen):
        for tok in sen:
            if tok in self.keys:
                self.k += 1
        self.n += len(sen)

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.k / self.n

    def show(self):
        return '%.4f' % (1. * self.value())


class SelectFrequencyMetric(object):
    def __init__(self):
        self.min_color = 53
        self.max_color = 203
        self.min_size = 7
        self.max_size = 13
        self.color_bin = 5
        self.color_range = 1 + int((self.max_color - self.min_color) / self.color_bin)
        self.size_range = self.max_size - self.min_size + 1
        self.total_color_size = np.zeros((self.color_range, self.size_range))
        self.selected_color_size = np.zeros((self.color_range, self.size_range))
        self.selected_x_value = np.array([])
        self.selected_y_value = np.array([])

    def _rgb_to_int(self, color):
        return int(re.search(r"[\d]+", color).group(0))

    def _group_color(self, color):
        return int((self._rgb_to_int(color) - self.min_color) / self.color_bin)

    def _group_size(self, size):
        return size - self.min_size

    def reset(self):
        pass

    def record(self, selected_obj, kb):
        size = self._group_size(selected_obj['size'])
        color = self._group_color(selected_obj['color'])
        self.selected_color_size[color][size] += 1
        for obj in kb:
            size = self._group_size(obj['size'])
            color = self._group_color(obj['color'])
            self.total_color_size[color][size] += 1
        self.selected_x_value = np.append(self.selected_x_value, selected_obj['x'])
        self.selected_y_value = np.append(self.selected_y_value, selected_obj['y'])

    def plot(self, name):
        ax = sns.heatmap(np.divide(self.selected_color_size, self.total_color_size,
                        out=np.zeros_like(self.selected_color_size), where=self.total_color_size!=0),
                        cmap=cm.Blues, yticklabels=3)
        plt.xlabel('size', fontsize=18)
        plt.ylabel('color', fontsize=18)
        xticklabels = [str(int(x.get_text()) + self.min_size) for x in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels)
        yticklabels = [str(int(y.get_text())*self.color_bin + self.min_color) for y in ax.get_yticklabels()]
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.savefig(name + '_size_color.png', dpi=300)
        plt.clf()
        ax = sns.scatterplot(x=self.selected_x_value, y=self.selected_y_value, size=0, legend=False)
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        plt.axes().set_aspect('equal', 'datalim')
        plt.savefig(name + '_location.png', dpi=300)
        plt.clf()

class MetricsContainer(object):
    def __init__(self):
        self.metrics = OrderedDict()
        self.plot_metrics = OrderedDict()

    def _register(self, name, ty, *args, **kwargs):
        name = name.lower()
        assert name not in self.metrics
        self.metrics[name] = ty(*args, **kwargs)

    def _register_plot(self, name, ty, *args, **kwargs):
        name = name.lower()
        assert name not in self.plot_metrics
        self.plot_metrics[name] = ty(*args, **kwargs)

    def register_average(self, name, *args, **kwargs):
        self._register(name, AverageMetric, *args, **kwargs)

    def register_moving_average(self, name, *args, **kwargs):
        self._register(name, MovingAverageMetric, *args, **kwargs)

    def register_time(self, name, *args, **kwargs):
        self._register(name, TimeMetric, *args, **kwargs)

    def register_percentage(self, name, *args, **kwargs):
        self._register(name, PercentageMetric, *args, **kwargs)

    def register_moving_percentage(self, name, *args, **kwargs):
        self._register(name, MovingPercentageMetric, *args, **kwargs)

    def register_ngram(self, name, *args, **kwargs):
        self._register(name, NGramMetric, *args, **kwargs)

    def register_similarity(self, name, *args, **kwargs):
        self._register(name, SimilarityMetric, *args, **kwargs)

    def register_uniqueness(self, name, *args, **kwargs):
        self._register(name, UniquenessMetric, *args, **kwargs)

    def register_word_frequency(self, name, *args, **kwargs):
        self._register(name, WordFrequencyMetric, *args, **kwargs)

    def register_select_frequency(self, name, *args, **kwargs):
        self._register_plot(name, SelectFrequencyMetric, *args, **kwargs)

    def record(self, name, *args, **kwargs):
        name = name.lower()
        if name in self.metrics:
            self.metrics[name].record(*args, **kwargs)
        elif name in self.plot_metrics:
            self.plot_metrics[name].record(*args, **kwargs)
        else:
            assert False

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def value(self, name):
        return self.metrics[name].value()

    def show(self):
        return ' '.join(['%s=%s' % (k, v.show()) for k, v in self.metrics.items()])

    def plot(self):
        for k, v in self.plot_metrics.items():
            v.plot(k)

    def dict(self):
        d = OrderedDict()
        for k, v in self.metrics.items():
            d[k] = v.show()
        return d