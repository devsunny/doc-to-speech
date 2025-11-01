import pathlib
import click

@click.group
def main():
    """document to speech CLI interface"""
    pass

@click.command("wav")
@click.option("--source", "-s",  type=click.Path(path_type=pathlib.Path, exists=True, readable=True, dir_okay=True, file_okay=True), help="Path to the source document or directory.")
@click.option("--output", "-o",  type=click.Path(path_type=pathlib.Path), help="Path to the output WAV file.")
def document_to_wav(source, output):
    """Convert document(s) to WAV format."""
    from dts.speaker import DocumentToSpeech

    dts = DocumentToSpeech()
    source_path = pathlib.Path(source)

    if source_path.is_file():
        output_wav = output.with_suffix(".wav")
        dts.doc_to_wav(source_path, output_wav_path=output_wav)
        click.echo(f"Converted {source_path} to {output_wav}")
    elif source_path.is_dir():
        for doc_file in source_path.rglob("*"):
            if doc_file.is_file():
                output_wav = doc_file.with_suffix(".wav")
                dts.doc_to_wav(doc_file, output_wav_path=output_wav)
                click.echo(f"Converted {doc_file} to {output_wav}")

@click.command("mp3")
@click.option("--source", "-s",  type=click.Path(path_type=pathlib.Path, exists=True, readable=True, dir_okay=True, file_okay=True), help="Path to the source document or directory.")
@click.option("--output", "-o",  type=click.Path(path_type=pathlib.Path), help="Path to the output MP3 file.")
def document_to_mp3(source, output):
    """Convert document(s) to MP3 format."""
    from dts.speaker import DocumentToSpeech

    dts = DocumentToSpeech()
    source_path = pathlib.Path(source)

    if source_path.is_file():
        output_mp3 = output.with_suffix(".mp3")
        dts.doc_to_mp3(source_path, output_mp3_path=output_mp3)
        click.echo(f"Converted {source_path} to {output_mp3}")
    elif source_path.is_dir():
        for doc_file in source_path.rglob("*"):
            if doc_file.is_file():
                output_mp3 = doc_file.with_suffix(".mp3")
                dts.doc_to_mp3(doc_file, output_mp3_path=output_mp3)
                click.echo(f"Converted {doc_file} to {output_mp3}")

@click.command("wav2mp3")
@click.option("--source", "-s",  type=click.Path(path_type=pathlib.Path, exists=True, readable=True, dir_okay=True, file_okay=True), help="Path to the source document or directory.")
@click.option("--output", "-o",  type=click.Path(path_type=pathlib.Path), help="Path to the output MP3 file.")
def wav_to_mp3(source, output):
    """Convert wav files to MP3 format."""
    from pydub import AudioSegment
    
    source_path = pathlib.Path(source)
    if source_path.is_file():
        sound = AudioSegment.from_wav(source_path)
        output_mp3 = output.with_suffix(".mp3")
        sound.export(output_mp3, format="mp3")
        click.echo(f"Converted {source_path} to {output_mp3}")
    elif source_path.is_dir():
        for wav_file in source_path.rglob("*.wav"):
            if wav_file.is_file():
                sound = AudioSegment.from_wav(wav_file)
                output_mp3 = wav_file.with_suffix(".mp3")
                sound.export(output_mp3, format="mp3")
                click.echo(f"Converted {wav_file} to {output_mp3}")    
    
main.add_command(document_to_mp3)
main.add_command(document_to_wav)
main.add_command(wav_to_mp3)

if __name__ == "__main__":
    main()
