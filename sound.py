import os

import pygame

class Sounds:
    def __init__(self):
        self.sounds = []
        self.musics = []
        sound_folder_path = os.path.normpath(os.path.join(os.path.abspath(__file__),  "../sound"))
        se0 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou09.mp3"))      
        se1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou47.wav"))
        se2 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou41.wav"))
        se3 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou48.wav"))
        se4 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou19.wav"))
        se0.set_volume(0.3)
        se1.set_volume(0.3)
        se2.set_volume(0.3)
        se3.set_volume(0.3)
        se4.set_volume(0.3)

        self.sounds.append(se0)
        self.sounds.append(se1)
        self.sounds.append(se2)
        self.sounds.append(se3)
        self.sounds.append(se4)

        bgm1 = os.path.join(sound_folder_path, "maou09.mp3")
        bgm2 = os.path.join(sound_folder_path, "tamsu08.mp3")

        self.musics.append(bgm1)
        self.musics.append(bgm2)
        
        self.bgm_play(0)

    def bgm_play(self, id, loop=-1):
        if id<0 or id>=len(self.musics):
            return
        pygame.mixer.music.load(self.musics[id])
        pygame.mixer.music.play(loop)

    def play(self, id, loop=0):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].play(loops=loop)
    
    def stop(self, id):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].stop()
