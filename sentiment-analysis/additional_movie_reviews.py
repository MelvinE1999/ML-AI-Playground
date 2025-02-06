# just to store this variable to be called within the sentiment-analysis.py file as messing around with adding data to already provided data
additional_movie_reviews = [
    ("9/10 - Absolutely mind-blowing cinematography! The visuals alone make this a must-watch.", "pos"),
    ("4/10 - The plot was all over the place, and I couldn't connect with any of the characters.", "neg"),
    ("A masterpiece in storytelling. The layers of symbolism left me thinking for days!", "pos"),
    ("5/10 - Not the worst movie I've seen, but it definitely felt like a missed opportunity.", "neg"),
    ("Pure brilliance! This film captures human emotion in a way few others have.", "pos"),
    ("2/10 - Boring, predictable, and honestly a waste of time.", "neg"),
    ("This film hit me in the soul. I’ll be thinking about it for weeks!", "pos"),
    ("8/10 - Solid performances, gripping plot, and a soundtrack that elevates everything.", "pos"),
    ("3/10 - The dialogue felt robotic, and the pacing was excruciatingly slow.", "neg"),
    ("A visual feast! Every frame felt like a painting come to life.", "pos"),
    ("10/10 - Simply put, a cinematic triumph!", "pos"),
    ("The movie was so profound that I had to sit in silence afterward to process it all.", "pos"),
    ("1/10 - Painfully bad. I wouldn't recommend this to my worst enemy.", "neg"),
    ("7/10 - Enjoyable but flawed. The ending could have been stronger.", "pos"),
    ("6/10 - Not great, not terrible. Just another average flick.", "neg"),
    ("Deeply emotional and beautifully acted. I was hooked from start to finish.", "pos"),
    ("0/10 - I regret every second I spent watching this trainwreck.", "neg"),
    ("Thought-provoking and profound. The themes resonate long after the credits roll.", "pos"),
    ("A true rollercoaster of emotions. I laughed, I cried, I cheered.", "pos"),
    ("Too pretentious for its own good. Feels like it’s trying way too hard.", "neg"),
    ("A 10/10 experience! I was completely immersed from the first frame.", "pos"),
    ("4/10 - The action was decent, but everything else fell flat.", "neg"),
    ("Incredible pacing and an unforgettable ending. This one will stay with me.", "pos"),
    ("1/10 - Who greenlit this disaster?", "neg"),
    ("9/10 - I haven’t seen a film this captivating in years!", "pos"),
    ("The writing is so sharp, every line of dialogue hits perfectly.", "pos"),
    ("3/10 - It had potential, but the execution was painfully bad.", "neg"),
    ("A film that understands human nature like no other. Stunningly insightful.", "pos"),
    ("An utter snoozefest. I nearly fell asleep three times.", "neg"),
    ("One of the most original films of the decade! The creativity is off the charts.", "pos"),
    ("2/10 - Felt like an amateur film school project, not a professional production.", "neg"),
    ("If you love intelligent cinema, this is a must-see. Mind-blowing concepts!", "pos"),
    ("The cinematography alone is worth watching this film for. Absolutely stunning.", "pos"),
    ("An emotional gut punch. I haven’t cried like this over a movie in years.", "pos"),
    ("Such a deeply layered narrative. I’m still piecing it together.", "pos"),
    ("Overrated. I don’t see why critics are raving about it.", "neg"),
    ("A near-perfect film. Everything just works so seamlessly together.", "pos"),
    ("Terrible pacing, cringeworthy dialogue, and an ending that made no sense.", "neg"),
    ("A brilliant commentary on society wrapped in a beautifully shot movie.", "pos"),
    ("Unforgettable performances that elevate an already excellent script.", "pos"),
    ("A thought-provoking, mind-expanding journey. Movies like this are rare gems.", "pos"),
    ("It felt like a two-hour slog. Nothing happens, and it ends on a whimper.", "neg"),
    ("It changed the way I see the world. Profoundly moving.", "pos"),
    ("This film speaks to the soul. Absolutely beautiful storytelling.", "pos"),
    ("Laughably bad. I expected better from the director.", "neg"),
    ("9/10 - A modern classic in the making. I’ll be rewatching this for years.", "pos"),
    ("6/10 - Some good moments, but overall, nothing to write home about.", "neg"),
    ("I walked out of the theater in awe. This is what cinema is all about.", "pos"),
    ("A fantastic blend of style and substance. Rarely do we get both in one film.", "pos"),
    ("Holy shit, this movie was next-level amazing! Easily a 10/10.", "pos"),
    ("5/10 - Meh. It wasn’t the worst thing ever, but I expected way more.", "neg"),
    ("Absolute garbage. What the hell were they thinking? 1/10.", "neg"),
    ("Freakin’ awesome! This movie blew my damn mind.", "pos"),
    ("Yo, this flick slapped hard. I’m still thinking about it! 9/10.", "pos"),
    ("What a load of crap. I’d rather watch paint dry. 2/10.", "neg"),
    ("Damn, that twist was wild! Totally didn’t see that coming. 8/10.", "pos"),
    ("This movie was dumb as hell. Save yourself the trouble. 1/10.", "neg"),
    ("Bro, that ending hit different. I need a moment. 9.5/10.", "pos"),
    ("This was straight-up trash. I feel robbed. 0/10.", "neg"),
    ("Insane action scenes! My heart was racing the whole time. 9/10.", "pos"),
    ("A total letdown. Thought it would be epic, but nope. 3/10.", "neg"),
    ("Shit, this movie had me on the edge of my seat! 9/10.", "pos"),
    ("Yo, this was next-level bad. I can’t believe I wasted two hours on this. 1/10.", "neg")
]