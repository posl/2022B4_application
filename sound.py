import os

import pygame

class Sounds:
    def __init__(self):
        self.sounds = []
        sound_folder_path = os.path.normpath(os.path.join(os.path.abspath(__file__),  "../sound"))
        bgm1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou09.mp3"))
        self.sounds.append(bgm1)
        bgm1.play(loops=-1)

        se1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou47.wav"))
        se2 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou41.wav"))
        se3 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou48.wav"))
        se4 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou19.wav"))

        self.sounds.append(se1)
        self.sounds.append(se2)
        self.sounds.append(se3)
        self.sounds.append(se4)

    def play(self, id, loop=0):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].play(loops=loop)
    
    def stop(self, id):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].stop()
