def get_system(name, args, schema=None, timed=False, model_path=None):
    #if name in ('sv_model', 'rl_model'):
        #lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words, lexicon_path=args.lexicon)
        #if args.inverse_lexicon:
        #    realizer = InverseLexicon.from_file(args.inverse_lexicon)
        #else:
        #    realizer = DefaultInverseLexicon()
    if name == 'yourmodel':
        raise ValueError('System not defined yet')
    else:
        raise ValueError('Unknown system %s' % name)