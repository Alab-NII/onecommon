import re
import pdb

def get_domain(name):
    """Creates domain by name."""
    if name == 'object_division':
        return ObjectDivisionDomain()
    elif name == 'one_common':
        return OneCommonDomain()
    raise()

class Domain(object):
    """Domain interface."""
    def selection_length(self):
        """The length of the selection output."""
        pass

    def input_length(self):
        """The length of the context/input."""
        pass

    def generate_choices(self, ctx):
        """Generates all the possible valid choices based on the given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def parse_context(self, ctx):
        """Parses a given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def score(self, context, choice):
        """Scores the dialogue.

        context: the input of the dialogue.
        choice: the generated choice by an agent.
        """
        pass

    def parse_choice(self, choice):
        """Parses the generated choice.

        choice: a list of strings like 'itemX=Y'
        """
        pass

    def parse_human_choice(self, inpt, choice):
        """Parses human choices. It has extra validation that parse_choice.

        inpt: the context of the dialogue.
        choice: the generated choice by a human
        """
        pass

    def score_choices(self, choices, ctxs):
        """Scores choices.

        choices: agents choices.
        ctxs: agents contexes.
        """
        pass

class OneCommonDomain(Domain):
    """Instance of the one common domain."""
    def selection_length(self):
        return 1

    def input_length(self):
        return 28

    def num_ent(self):
        return 7

    def dim_ent(self):
        return 4
