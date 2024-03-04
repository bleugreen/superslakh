import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel
from openai import OpenAI

vertexai.init(project="nimbus-404702", location="us-central1")
parameters = {
    "max_output_tokens": 256,
    "temperature": 0.1,
    "top_p": 1
}

def predict_gm_class_gemini(sf):
    model = GenerativeModel("gemini-pro")
    chat = model.start_chat(response_validation=False)

    response = chat.send_message(
        """Use the following MIDI Instrument class list to classify the given soundfont preset. It must match one instrument from this list exactly (program, name):

    program,name
    1,Acoustic Grand Piano
    2,Bright Acoustic Piano
    3,Electric Grand Piano
    4,Honky-tonk Piano
    5,Electric Piano 1
    6,Electric Piano 2
    7,Harpsichord
    8,Clavinet
    9,Celesta
    10,Glockenspiel
    11,Music Box
    12,Vibraphone
    13,Marimba
    14,Xylophone
    15,Tubular Bells
    16,Dulcimer
    17,Drawbar Organ
    18,Percussive Organ
    19,Rock Organ
    20,Church Organ
    21,Reed Organ
    22,Accordion
    23,Harmonica
    24,Tango Accordion
    25,Acoustic Guitar (nylon)
    26,Acoustic Guitar (steel)
    27,Electric Guitar (jazz)
    28,Electric Guitar (clean)
    29,Electric Guitar (muted)
    30,Overdriven Guitar
    31,Distortion Guitar
    32,Guitar harmonics
    33,Acoustic Bass
    34,Electric Bass (finger)
    35,Electric Bass (pick)
    36,Fretless Bass
    37,Slap Bass 1
    38,Slap Bass 2
    39,Synth Bass 1
    40,Synth Bass 2
    41,Violin
    42,Viola
    43,Cello
    44,Contrabass
    45,Tremolo Strings
    46,Pizzicato Strings
    47,Orchestral Harp
    48,Timpani
    49,String Ensemble 1
    50,String Ensemble 2
    51,SynthStrings 1
    52,SynthStrings 2
    53,Choir Aahs
    54,Voice Oohs
    55,Synth Voice
    56,Orchestra Hit
    57,Trumpet
    58,Trombone
    59,Tuba
    60,Muted Trumpet
    61,French Horn
    62,Brass Section
    63,SynthBrass 1
    64,SynthBrass 2
    65,Soprano Sax
    66,Alto Sax
    67,Tenor Sax
    68,Baritone Sax
    69,Oboe
    70,English Horn
    71,Bassoon
    72,Clarinet
    73,Piccolo
    74,Flute
    75,Recorder
    76,Pan Flute
    77,Blown Bottle
    78,Shakuhachi
    79,Whistle
    80,Ocarina
    81,Lead 1 (square)
    82,Lead 2 (sawtooth)
    83,Lead 3 (calliope)
    84,Lead 4 (chiff)
    85,Lead 5 (charang)
    86,Lead 6 (voice)
    87,Lead 7 (fifths)
    88,Lead 8 (bass + lead)
    89,Pad 1 (new age)
    90,Pad 2 (warm)
    91,Pad 3 (polysynth)
    92,Pad 4 (choir)
    93,Pad 5 (bowed)
    94,Pad 6 (metallic)
    95,Pad 7 (halo)
    96,Pad 8 (sweep)
    97,FX 1 (rain)
    98,FX 2 (soundtrack)
    99,FX 3 (crystal)
    100,FX 4 (atmosphere)
    101,FX 5 (brightness)
    102,FX 6 (goblins)
    103,FX 7 (echoes)
    104,FX 8 (sci-fi)
    105,Sitar
    106,Banjo
    107,Shamisen
    108,Koto
    109,Kalimba
    110,Bagpipe
    111,Fiddle
    112,Shanai
    113,Tinkle Bell
    114,Agogo
    115,Steel Drums
    116,Woodblock
    117,Taiko Drum
    118,Melodic Tom
    119,Synth Drum
    120,Reverse Cymbal
    121,Guitar Fret Noise
    122,Breath Noise
    123,Seashore
    124,Bird Tweet
    125,Telephone Ring
    126,Helicopter
    127,Applause
    128,Gunshot

    input: {\'name\': \'Spudge Bass\', \'bank\': 0, \'preset\': 73, \'sf_path\': \'141-compleet_bank__synth.sf2\'}
    output: 39,Synth Bass 1

    input: {\'name\': \'Muted Steel Guitar\', \'bank\': 1, \'preset\': 25, \'sf_path\': \"Stgiga\'s HiDef Soundfont (2019-05-25).sf2\"}
    output: 26,Acoustic Guitar (steel)

    input: {\'name\': \'024:052 Chorus Lahs\', \'bank\': 24, \'preset\': 52, \'sf_path\': \"Stgiga\'s HiDef Soundfont (2019-05-25).sf2\"}
    output: 53,Choir Aahs

    input: {\'name\': \'Pinkullo en Do seco\', \'bank\': 0, \'preset\': 0, \'sf_path\': \'132-pinkullo2.sf2\'}
    output: 80,Ocarina

    input: {\'name\': \'Jazz Man\', \'bank\': 45, \'preset\': 28, \'sf_path\': \'OmegaGMGS2.sf2\'}
    output: 27,Electric Guitar (jazz)

    """+f"\ninput:{sf}\noutput:")
    return response.text


def predict_gm_class_openai(sf):
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "Use the following MIDI Instrument class list to classify the given soundfont preset. It must match one instrument from this list exactly (`program, name`). If the instrument is not defined in the list, choose the closest option:\n\n    program,name\n    1,Acoustic Grand Piano\n    2,Bright Acoustic Piano\n    3,Electric Grand Piano\n    4,Honky-tonk Piano\n    5,Electric Piano 1\n    6,Electric Piano 2\n    7,Harpsichord\n    8,Clavinet\n    9,Celesta\n    10,Glockenspiel\n    11,Music Box\n    12,Vibraphone\n    13,Marimba\n    14,Xylophone\n    15,Tubular Bells\n    16,Dulcimer\n    17,Drawbar Organ\n    18,Percussive Organ\n    19,Rock Organ\n    20,Church Organ\n    21,Reed Organ\n    22,Accordion\n    23,Harmonica\n    24,Tango Accordion\n    25,Acoustic Guitar (nylon)\n    26,Acoustic Guitar (steel)\n    27,Electric Guitar (jazz)\n    28,Electric Guitar (clean)\n    29,Electric Guitar (muted)\n    30,Overdriven Guitar\n    31,Distortion Guitar\n    32,Guitar harmonics\n    33,Acoustic Bass\n    34,Electric Bass (finger)\n    35,Electric Bass (pick)\n    36,Fretless Bass\n    37,Slap Bass 1\n    38,Slap Bass 2\n    39,Synth Bass 1\n    40,Synth Bass 2\n    41,Violin\n    42,Viola\n    43,Cello\n    44,Contrabass\n    45,Tremolo Strings\n    46,Pizzicato Strings\n    47,Orchestral Harp\n    48,Timpani\n    49,String Ensemble 1\n    50,String Ensemble 2\n    51,SynthStrings 1\n    52,SynthStrings 2\n    53,Choir Aahs\n    54,Voice Oohs\n    55,Synth Voice\n    56,Orchestra Hit\n    57,Trumpet\n    58,Trombone\n    59,Tuba\n    60,Muted Trumpet\n    61,French Horn\n    62,Brass Section\n    63,SynthBrass 1\n    64,SynthBrass 2\n    65,Soprano Sax\n    66,Alto Sax\n    67,Tenor Sax\n    68,Baritone Sax\n    69,Oboe\n    70,English Horn\n    71,Bassoon\n    72,Clarinet\n    73,Piccolo\n    74,Flute\n    75,Recorder\n    76,Pan Flute\n    77,Blown Bottle\n    78,Shakuhachi\n    79,Whistle\n    80,Ocarina\n    81,Lead 1 (square)\n    82,Lead 2 (sawtooth)\n    83,Lead 3 (calliope)\n    84,Lead 4 (chiff)\n    85,Lead 5 (charang)\n    86,Lead 6 (voice)\n    87,Lead 7 (fifths)\n    88,Lead 8 (bass + lead)\n    89,Pad 1 (new age)\n    90,Pad 2 (warm)\n    91,Pad 3 (polysynth)\n    92,Pad 4 (choir)\n    93,Pad 5 (bowed)\n    94,Pad 6 (metallic)\n    95,Pad 7 (halo)\n    96,Pad 8 (sweep)\n    97,FX 1 (rain)\n    98,FX 2 (soundtrack)\n    99,FX 3 (crystal)\n    100,FX 4 (atmosphere)\n    101,FX 5 (brightness)\n    102,FX 6 (goblins)\n    103,FX 7 (echoes)\n    104,FX 8 (sci-fi)\n    105,Sitar\n    106,Banjo\n    107,Shamisen\n    108,Koto\n    109,Kalimba\n    110,Bagpipe\n    111,Fiddle\n    112,Shanai\n    113,Tinkle Bell\n    114,Agogo\n    115,Steel Drums\n    116,Woodblock\n    117,Taiko Drum\n    118,Melodic Tom\n    119,Synth Drum\n    120,Reverse Cymbal\n    121,Guitar Fret Noise\n    122,Breath Noise\n    123,Seashore\n    124,Bird Tweet\n    125,Telephone Ring\n    126,Helicopter\n    127,Applause\n    128,Gunshot\n\nRemember, you can only respond with members of this list, NOTHING ELSE"
        },
        {
        "role": "user",
        "content": "{\\'name\\': \\'Spudge Bass\\', \\'bank\\': 0, \\'preset\\': 73, \\'sf_path\\': \\'141-compleet_bank__synth.sf2\\'}"
        },
        {
        "role": "assistant",
        "content": "`39,Synth Bass 1`"
        },
        {
        "role": "user",
        "content": "{\\'name\\': \\'Muted Steel Guitar\\', \\'bank\\': 1, \\'preset\\': 25, \\'sf_path\\': \\\"Stgiga\\'s HiDef Soundfont (2019-05-25).sf2\\\"}"
        },
        {
        "role": "assistant",
        "content": "`26,Acoustic Guitar (steel)`"
        },
        {
        "role": "user",
        "content": "{\\'name\\': \\'024:052 Chorus Lahs\\', \\'bank\\': 24, \\'preset\\': 52, \\'sf_path\\': \\\"Stgiga\\'s HiDef Soundfont (2019-05-25).sf2\\\"}"
        },
        {
        "role": "assistant",
        "content": "`53,Choir Aahs`"
        },
        {
        "role": "user",
        "content": "{\\'name\\': \\'Pinkullo en Do seco\\', \\'bank\\': 0, \\'preset\\': 0, \\'sf_path\\': \\'132-pinkullo2.sf2\\'}"
        },
        {
        "role": "assistant",
        "content": "`74,Flute`"
        },
        {
        "role":"user",
        "content":str(sf)
        }
    ],
    temperature=0.22,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].message.content