import sys
import os
from collections import defaultdict

def read_segments(segment_file):
    """ Read segment file and return a dictionary {audio_filename: {segment_id: (start_time, end_time)}} """
    segments = defaultdict(dict)

    with open(segment_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(f"Error in segment file: Expected 4 fields, got {len(parts)}.\nLine: {line.strip()}")

            segment_id, audio_filename, start_time, end_time = parts
            segments[audio_filename][segment_id] = (float(start_time), float(end_time))
    
    return segments

def read_transcriptions(transcription_file):
    """ Read ASR transcriptions and return a dictionary {segment_id: text} """
    transcriptions = {}
    
    with open(transcription_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            segment_id, text = parts
            transcriptions[segment_id] = text
    
    return transcriptions

def generate_trs(output_dir, audio_filename, segments, transcriptions):
    """ Generate a .trs file for a single audio file """
    trs_filename = os.path.join(output_dir, f"{audio_filename}.trs")
    sorted_segments = sorted(segments.items(), key=lambda x: x[1][0])  # Sort by start time
    startTime = min([v[0] for v in segments.values()])
    endTime = max([v[1] for v in segments.values()])  # Compute max end time

    with open(trs_filename, 'w') as f:
        # Write header
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<!DOCTYPE Trans SYSTEM "trans-14.dtd">\n')
        f.write(f'<Trans scribe="auto" audio_filename="{audio_filename}" version="2" version_date="2025-02-17">\n')
        f.write('<Episode>\n')
        f.write(f'<Section type="report" startTime="{startTime:.2f}" endTime="{endTime:.2f}">\n')
        f.write(f'<Turn startTime="{startTime:.2f}" endTime="{endTime:.2f}">\n')

        # Write Sync tags with transcriptions
        for segment_id, (start_time, _) in sorted_segments:
            transcription = transcriptions.get(segment_id, "[UNK]")  # Default to [UNK] if missing
            f.write(f'  <Sync time="{start_time:.2f}"/> {transcription}\n')

        # Close tags
        f.write('</Turn>\n')
        f.write('</Section>\n')
        f.write('</Episode>\n')
        f.write('</Trans>\n')

    #print(f"Saved TRS: {trs_filename}")

def main(segment_file, transcription_file, output_dir):
    """ Main function to convert segment and transcription files to multiple .trs files """
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        segments_by_audio = read_segments(segment_file)
        transcriptions = read_transcriptions(transcription_file)

        # Generate a .trs file for each unique audio file
        for audio_filename, segments in segments_by_audio.items():
            generate_trs(output_dir, audio_filename, segments, transcriptions)
        print(f"Saved all .trs files in {output_dir}") 
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_to_trs.py <segment_file> <transcription_file> <output_directory>")
        sys.exit(1)
    
    segment_file = sys.argv[1]
    transcription_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    main(segment_file, transcription_file, output_dir)

