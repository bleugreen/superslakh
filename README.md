# superslakh - a large synthetic music dataset

### what is it?
- 105,057 midi songs, with over a million instrument stems (subset of lakh midi)
- ~9,000 instrument presets (containing synths & sampled instruments)
- a classification / rendering process to combine the two

### why?
it can be used to train pretty much any music task

you have the 'source code' of the audio, making it simple to augment / modify the data, all in a format which is dramatically smaller than even the most compressed audio

each midi file contains:
- local tempo
- time signature
- key signature
- score ('piano roll') for each instrument
	- 128 instrument classes ([General Midi](https://en.wikipedia.org/wiki/General_MIDI))
	- ~8 instruments per song on average
- many even have lyrics
	- *todo - render vocals with SVS using lyrics / melody*

this makes it well-suited to tasks like:
- tempo / key classification
- beat / downbeat prediction
- stem separation
- multi-instrument transcription
- generative composition ('write a bass line for this piano melody')

### references
[lakh midi dataset](https://colinraffel.com/projects/lmd/) - where all the midi is from (s/o colin raffel)
[pretty-midi](https://github.com/craffel/pretty-midi)- how the midi files are handled & rendered (also by colin raffel)
[slakh2100](http://www.slakh.com/) - where I got the idea