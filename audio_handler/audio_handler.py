import moviepy.editor as mp

def addAudioFiles(videoPath, audioList, output_path):
    videoFile = mp.VideoFileClip(videoPath)
    videoClips = []

    for i in range(len(audioList)):
        if i < 1:
            end = audioList[i]['start_in']
            clip = videoFile.subclip(0, end)
            videoClips.append(clip)
        else:
            start_in = audioList[i - 1]['start_in']
            end_in = audioList[i]['start_in']
            audio = mp.AudioFileClip(audioList[i - 1]['audio_path'])
            clip = videoFile.subclip(start_in, end_in).set_audio(audio)
            videoClips.append(clip)

    # Final segment from the last audio to the end of the video
    start_in = audioList[-1]['start_in']
    audio = mp.AudioFileClip(audioList[-1]['audio_path'])
    clip = videoFile.subclip(start_in, videoFile.duration).set_audio(audio)
    videoClips.append(clip)
    
    finalVideo = mp.concatenate_videoclips(videoClips)

    # Specify the codec. For `.avi`, you can try 'png' or 'libx264'
    finalVideo.write_videofile(output_path, codec='libx264')

