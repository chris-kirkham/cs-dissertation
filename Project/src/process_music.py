#Script which uses pretty-midi to process user-input music for use in the system's neural networks

import pretty_midi #http://craffel.github.io/pretty-midi/
import itertools
import sys
import numpy

#PROCESS MUSIC
timeScalar = 20 #precision of 50ms

def processMusic(midi, max_sequence_timesteps):
    midiPM = pretty_midi.PrettyMIDI(midi)
    
    #get notes from midi file
    notes = list()
    
    for instrument in midiPM.instruments:
        notes.append(instrument.notes)
        
    #flatten instruments so all notes are in 1D list; sort by start time
    notes = sorted(list(itertools.chain.from_iterable(notes)), key = lambda x:x.start) 
    
    #multiply note times by timeScalar
    for note in notes:
        note.start *= timeScalar
        note.end *= timeScalar

    #round notes to nearest integer
    for note in notes:
        note.start = int(round(note.start))
        note.end = int(round(note.end))

    #process notes to array of [timesteps, pitches]
    musicProcessed = numpy.zeros((max_sequence_timesteps, 128))
    for note in notes:
        for timestep in range(round(note.start), round(note.end)):
            if(timestep < max_sequence_timesteps):
                musicProcessed[timestep][note.pitch] = note.velocity / 128
            
    return musicProcessed