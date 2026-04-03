def generate_srt(word_timestamps, time_stride, words_per_subtitle=5):
    """Generates SRT formatted string from word timestamps."""
    srt_lines = []
    sub_index = 1
    
    for i in range(0, len(word_timestamps), words_per_subtitle):
        chunk = word_timestamps[i:i + words_per_subtitle]
        if not chunk:
            continue
            
        first_word = chunk[0]
        start_time = first_word.get('start_time', first_word.get('start_offset', 0) * time_stride)
        
        last_word = chunk[-1]
        end_time = last_word.get('end_time', last_word.get('end_offset', 0) * time_stride)
        
        text = " ".join([w.get('word', w.get('char', '')) for w in chunk])
        
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int(round((seconds - int(seconds)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
        srt_lines.append(f"{sub_index}")
        srt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
        srt_lines.append(f"{text}\n")
        
        sub_index += 1
        
    return "\n".join(srt_lines)