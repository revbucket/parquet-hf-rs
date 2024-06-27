use std::sync::Arc;
use std::time::Instant;
use std::io::BufRead;
use serde_json;
use serde_json::Value;
use anyhow::Error;
use clap::Parser;
use std::path::PathBuf;
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use arrow::array::{ArrayRef, StringBuilder, Float32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::basic::{Compression, ZstdLevel};
use regex::Regex;

pub mod s3;
pub mod io;




/*=================================================================
=                                  ARGS                           =
=================================================================*/



#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[arg(long, required=true, num_args=1..)]
    input: Vec<PathBuf>,

    #[arg(long, required=true)]
    output: PathBuf,

}


/*=================================================================
=                             UTILITIES.                          =
=================================================================*/

fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );

    pbar.inc(0);
    pbar
}


fn replace_extension(path: &PathBuf) -> PathBuf {
    let path = path.clone();
    let regex = Regex::new(r"\.jsonl?\.(?:zstd|gz)$").unwrap();
    let path_str = path.to_str().unwrap();
    
    let output_path = if regex.is_match(path_str) {
        let new_path = regex.replace(path_str, ".parquet");
        let path = PathBuf::from(new_path.into_owned());
        path 
    } else {
        path
    };
    output_path
}


fn _build_schema() -> arrow::datatypes::Schema {
    let schema: arrow::datatypes::Schema = Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("url", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("language_score", DataType::Float32, false),
        Field::new("fasttext_score", DataType::Float32, false)]);
    schema 
}

fn _find_max_item(json: Option<&Value>) -> Option<(&str, f64)> {
    if json.is_none() {
        return None;
    }
    let json = json.unwrap();
    json.as_object()?
        .iter()
        .filter_map(|(key, value)| {
            value.as_f64().map(|v| (key.as_str(), v))
        })
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
}

fn jsonl_to_parquet(input_path: &PathBuf, output_path: &PathBuf) -> Result<(), Error> {

    let contents = read_pathbuf_to_mem(input_path).unwrap();

    let mut text_builder = StringBuilder::new();
    let mut url_builder = StringBuilder::new();
    let mut id_builder = StringBuilder::new();
    let mut language_builder = StringBuilder::new();
    let mut language_score_builder = Float32Builder::new();
    let mut fasttext_score_builder = Float32Builder::new();

    for line in contents.lines() {
        let line = line.unwrap();
        let json : Value = serde_json::from_str(&line).unwrap();

        // MUST HAVE: text + url
        text_builder.append_value(json["text"].as_str().unwrap());
        url_builder.append_value(json["url"].as_str().unwrap());

        // Would like-to-have: id, language, language_score, fasttext_score
        id_builder.append_option(
            json.get("metadata")
                .and_then(|m| m.get("WARC-Record-ID"))
                .and_then(|v| v.as_str())
        );       
        let max_language_score = _find_max_item(json.get("language_id_whole_page_fasttext"));
        if max_language_score.is_none() {
            language_builder.append_option(None::<String>);
            language_score_builder.append_option(None);
        } else {
            let (lang_id, lang_score) = max_language_score.unwrap();
            language_builder.append_value(lang_id);
            language_score_builder.append_value(lang_score as f32);
        }

        let ft_score = json.get("fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob");
        if ft_score.is_none() {
            fasttext_score_builder.append_option(None);
        } else {
            fasttext_score_builder.append_value(ft_score.unwrap().as_f64().unwrap() as f32);
        }
    }

    let text_array : ArrayRef = Arc::new(text_builder.finish());
    let url_array : ArrayRef = Arc::new(url_builder.finish());
    let id_array : ArrayRef = Arc::new(id_builder.finish());
    let language_array : ArrayRef = Arc::new(language_builder.finish());
    let language_score_array : ArrayRef = Arc::new(language_score_builder.finish());
    let fasttext_score_array : ArrayRef = Arc::new(fasttext_score_builder.finish());


    let schema = _build_schema();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![text_array, url_array, id_array, language_array, language_score_array, fasttext_score_array],
    )?;    

    let mut buf = Vec::new();
    {
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))  // Use zstd compression
            .build();        
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::new(schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
    }    

    write_mem_to_pathbuf(&buf, output_path).unwrap();
    Ok(())
}



/*=================================================================
=                                  MAIN                           =
=================================================================*/

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();

    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let pbar = build_pbar(paths.len(), "Paths");

    paths.par_iter()
        .for_each(|p| {
            let output_path = get_output_filename(&args.input, p, &args.output);
            let output_path = replace_extension(&output_path);
            jsonl_to_parquet(p, &output_path).unwrap();
            pbar.inc(1);
        });


    println!("-------------------------");
    println!("Finishing parquet creation in {:?} seconds", start_main.elapsed().as_secs());    
}