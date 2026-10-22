// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The captions sheet: the tray's Captions tool, as a form and then as a
//! progress report.
//!
//! One switch says what it captions: "Use Sample Voice", on, takes
//! the words from a clip with sound, chosen in the sheet from the clips on
//! the timeline that have any; off, from a script typed in it. It opens on
//! when a clip with sound is selected, or when a clip's menu opened it on
//! that clip, and off otherwise. Either lands as one batch of title clips,
//! one undo step.
//! The transcriber runs on a worker and reports as
//! [`CaptionsMsg::Progress`], ending as [`CaptionsMsg::Finished`].
//!
//! The sheet also reads a SubRip file in as captions, with the same look,
//! and writes the timeline's captions out as one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use concat_project::Command;
use concat_project::model::{ClipKey, ClipKind, KeyEase, KeyProperty, TextStyle};
use concat_project::srt;
use concat_speech::JobStage;
use concat_speech::transcribe::Segment;

use crate::host::{on_ui_in_project, spawn_in_project};
use crate::i18n::{t, tf};
use crate::panes::Msg;
use crate::panes::pace::{Pace, Stage, Timings};
use crate::panes::settings::installed;
use crate::studio::Studio;
use crate::ui::CaptionsSheetData;

/// Where a caption sits, by the sheet's row: a frame-height fraction from
/// the centre, positive down. Bottom, centre, top.
const CAPTION_OFFSETS: [f64; 3] = [0.35, 0.0, -0.35];
/// A caption's cap height by the sheet's row, as a fraction of the frame.
const CAPTION_SIZES: [f64; 3] = [0.04, 0.05, 0.065];
/// A rough speaking rate, for a script's timing and the speech sheet's
/// estimate.
pub const CHARS_PER_SECOND: f32 = 14.0;
/// The most words the sheet puts on screen at a time short of a phrase.
const MAX_WORDS: usize = 3;
/// A pause between two words longer than this ends a caption early, the
/// way a speaker's breath ends a line.
const PAUSE_SECONDS: f64 = 0.6;
/// A gap between two captions shorter than this is closed - the first is
/// held until the next comes in - so the words do not blink between them.
const HOLD_SECONDS: f64 = 0.4;
/// The shortest a caption of a few words is held, where the next leaves
/// room for it.
const MIN_WORDS_SECONDS: f64 = 0.25;

/// Everything that can happen to the captions sheet.
#[derive(Clone, Debug)]
pub enum CaptionsMsg {
    /// The tray's Captions tool: the selected clip's voice when it has
    /// sound, the script otherwise.
    Open,
    /// A clip's menu: the sheet on that clip's voice.
    OpenOnClip(String),
    Close,
    /// The "Use Sample Voice" switch.
    VoiceToggled(bool),
    /// Row in the list of clips with sound.
    SourceChanged(i32),
    TextEdited(String),
    ModelChanged(i32),
    PlacementChanged(i32),
    SizeChanged(i32),
    /// Words at a time, by the sheet's row: a phrase, then one to three.
    WordsChanged(i32),
    /// The entrance, by the sheet's row: none, pop, fade, rise.
    AnimationChanged(i32),
    /// "All caps".
    UpperToggled(bool),
    /// The look, by the sheet's row: the kept one, then each text preset.
    StyleChanged(i32),
    /// Run the pass the sheet describes.
    Begin,
    Cancel,
    /// The transcriber's worker reporting the stage it is in.
    Progress(JobStage),
    /// The transcriber's worker is done: what was said, or why not.
    Finished(Result<Vec<Segment>, String>),
    /// Asks for a SubRip file to read in.
    ImportSrt,
    /// The SubRip file chosen: its cues as captions.
    ImportSrtFrom(PathBuf),
    /// Asks where to write the captions as SubRip.
    ExportSrt,
    /// Where they go.
    ExportSrtTo(PathBuf),
}

/// The clip a transcription is running over, as the finished segments
/// need it: where it starts on the timeline and how fast it plays.
#[derive(Clone, Debug)]
struct Subject {
    start: f64,
    speed: f64,
    /// The look chosen when the run began.
    look: (f64, f64),
    /// How the words are laid out, as chosen when the run began.
    layout: Layout,
    /// The look, and the font it brings to register, as chosen when the
    /// run began.
    base: Option<TextStyle>,
    font: Option<Command>,
    /// The transcriber model, and the thirty-second windows whisper reads
    /// the sound in: what the run's timing is kept by.
    model: String,
    windows: f32,
}

/// Seconds whisper takes over one thirty-second window, per megabyte of
/// model, before a run on the model has been timed: about a second for
/// base and ten for medium on eight cores of a recent CPU.
const WINDOW_SECONDS_PER_MB: f32 = 1.0 / 160.0;

/// The captions sheet's state.
#[derive(Default)]
pub struct CaptionsPane {
    pub open: bool,
    /// "Use Sample Voice": the words come from [`Self::clip`]'s sound
    /// when on, from [`Self::text`] when off.
    pub from_voice: bool,
    /// The clip whose voice is transcribed: one with sound, chosen from the
    /// timeline's. Kept while the switch is off, so turning it back on
    /// returns to the same clip. The panel floats over a live editor, so
    /// a clip with sound selected since is the one shown and run instead;
    /// see [`Self::subject_clip`].
    pub clip: Option<String>,
    /// The sole selection when [`Self::clip`] was chosen - opened on, or
    /// picked from the list - so a later selection can be told from it.
    chosen_with: Option<String>,
    /// The script's words.
    pub text: String,
    /// Row in the installed transcriber list.
    pub model: usize,
    /// 0 bottom, 1 centre, 2 top.
    pub placement: usize,
    /// 0 small, 1 medium, 2 large.
    pub size: usize,
    /// Words on screen at a time: 0 a phrase, else 1..=3.
    pub words: usize,
    /// The entrance each caption makes; see [`Entrance`].
    pub animation: usize,
    /// Every caption in capitals.
    pub upper: bool,
    /// The look: 0 the kept one, else the text preset one before it in the
    /// studio's list.
    pub style: usize,
    pub running: bool,
    /// How far the running transcription looks.
    pub pace: Pace,
    /// Why the last run failed, when it did.
    pub message: String,
    /// The clip under the running transcription.
    subject: Option<Subject>,
    /// How long each model took this session, for the next run's bar.
    timings: Timings,
}

impl CaptionsPane {
    /// Applies one message. The studio is the rest of the window; while
    /// this runs the studio's copy of the pane is a blank it must not read.
    pub fn update(&mut self, msg: CaptionsMsg, studio: &mut Studio) {
        match msg {
            // A run in progress is shown as it is, however the panel is
            // asked for again: closing it never stopped the run.
            CaptionsMsg::Open | CaptionsMsg::OpenOnClip(_) if self.running => self.open = true,
            CaptionsMsg::Open => {
                let selected = studio.sole_selection();
                self.open_on(studio, selected, false);
            }
            CaptionsMsg::OpenOnClip(id) => self.open_on(studio, Some(id), true),
            CaptionsMsg::Close => self.open = false,
            CaptionsMsg::VoiceToggled(on) => {
                self.from_voice = on;
                self.message.clear();
                let sources = sources(studio);
                if on
                    && !self
                        .clip
                        .as_ref()
                        .is_some_and(|id| sources.iter().any(|s| &s.id == id))
                {
                    self.clip = sources.first().map(|source| source.id.clone());
                    self.chosen_with = studio.sole_selection();
                }
            }
            CaptionsMsg::SourceChanged(index) => {
                if let Some(source) = usize::try_from(index)
                    .ok()
                    .and_then(|index| sources(studio).into_iter().nth(index))
                {
                    self.clip = Some(source.id);
                    self.chosen_with = studio.sole_selection();
                    self.message.clear();
                }
            }
            CaptionsMsg::TextEdited(text) => self.text = text,
            CaptionsMsg::ModelChanged(index) => self.model = index.max(0) as usize,
            CaptionsMsg::PlacementChanged(index) => {
                self.placement = (index.max(0) as usize).min(2);
            }
            CaptionsMsg::SizeChanged(index) => self.size = (index.max(0) as usize).min(2),
            CaptionsMsg::WordsChanged(index) => {
                self.words = (index.max(0) as usize).min(MAX_WORDS);
                self.remember(studio);
            }
            CaptionsMsg::AnimationChanged(index) => {
                self.animation = (index.max(0) as usize).min(Entrance::ALL.len() - 1);
                self.remember(studio);
            }
            CaptionsMsg::UpperToggled(on) => {
                self.upper = on;
                self.remember(studio);
            }
            CaptionsMsg::StyleChanged(index) => {
                self.style = (index.max(0) as usize).min(studio.text_presets.len());
                self.remember(studio);
            }
            CaptionsMsg::Begin => {
                if self.from_voice {
                    self.run_sound(studio);
                } else {
                    self.run_script(studio);
                }
            }
            CaptionsMsg::Cancel => {
                studio.host.transcriber.cancel();
                self.running = false;
                self.open = false;
            }
            CaptionsMsg::Progress(stage) => self.pace.report(stage),
            CaptionsMsg::Finished(result) => {
                self.running = false;
                let subject = self.subject.take();
                if let (Ok(_), Some(subject)) = (&result, &subject) {
                    self.timings
                        .keep(&subject.model, self.pace.learned(subject.windows));
                }
                match result {
                    Ok(segments) => {
                        let Some(subject) = subject else {
                            return;
                        };
                        let mut commands =
                            spoken_captions(&segments, &subject, subject.base.as_ref());
                        let font = subject.font.clone();
                        let count = commands.len();
                        self.open = false;
                        if count == 0 {
                            studio.notify(&t("captions.nothingSaidClip"), true);
                        } else {
                            if let Some(font) = font {
                                commands.insert(0, font);
                            }
                            studio.apply(Command::Batch { commands });
                            studio.notify(&tf("captions.addedCaptions", &[&count]), false);
                        }
                    }
                    // Asked for: the sheet is already on its way down.
                    Err(error) if error.contains("cancel") => self.open = false,
                    // Closed while it ran: the panel is not there to say
                    // it, so the notice does, and the panel too next time.
                    Err(error) => {
                        if !self.open {
                            studio.notify(&error, true);
                        }
                        self.message = error;
                    }
                }
            }
            // Asked with the studio unborrowed, on a turn of its own; see
            // `host::after_dialog`.
            CaptionsMsg::ImportSrt => crate::host::pick_files_then(
                t("captions.importSrt"),
                Some((t("captions.subRipSubtitles"), &["srt"])),
                |studio, _, _, paths| {
                    if let Some(path) = paths.into_iter().next() {
                        studio.handle(Msg::Captions(CaptionsMsg::ImportSrtFrom(path)));
                    }
                },
            ),
            CaptionsMsg::ImportSrtFrom(path) => self.import_srt(studio, &path),
            CaptionsMsg::ExportSrt => self.ask_export_srt(studio),
            CaptionsMsg::ExportSrtTo(path) => self.export_srt(studio, &path),
        }
    }

    /// A SubRip file's cues as captions where the file times them, with
    /// the sheet's look: one batch, one undo step. A file that is not
    /// UTF-8 is read as best it can be.
    fn import_srt(&mut self, studio: &mut Studio, path: &Path) {
        let cues = std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                srt::parse(&String::from_utf8_lossy(&bytes)).map_err(|error| error.to_string())
            });
        let cues = match cues {
            Ok(cues) => cues,
            Err(error) => {
                log::warn!("captions: could not read {}: {error}", path.display());
                self.message = tf("captions.couldNotReadSubtitles", &[&error]);
                return;
            }
        };
        let look = self.look();
        let base = studio.prefs.title_style.clone();
        let commands = srt_commands(cues, look, base.as_ref());
        if commands.is_empty() {
            self.message = t("captions.noSubtitlesInFile");
            return;
        }
        let count = commands.len();
        self.open = false;
        studio.apply(Command::Batch { commands });
        studio.notify(&tf("captions.addedCaptions", &[&count]), false);
    }

    /// Asks where to write the captions, offering the project's folder
    /// and its name. A phone has no such dialog: the file is written
    /// beside the project under a name nothing there has, and handed to
    /// the phone to put where its file manager shows it.
    fn ask_export_srt(&mut self, studio: &mut Studio) {
        if export_lane(studio).is_none() {
            self.message = t("captions.noCaptionsToExport");
            return;
        }
        let name = format!("{}.srt", file_stem(studio));
        // No project, no project folder: Documents, as anywhere else a
        // file is written with nothing better to offer.
        let folder = studio
            .session
            .as_ref()
            .map(|session| PathBuf::from(session.path()))
            .unwrap_or_else(|| PathBuf::from(crate::studio::home_folder("Documents")));
        if cfg!(any(target_os = "android", target_os = "ios")) {
            let path = unique_path(&folder, &name);
            self.export_srt(studio, &path);
            return;
        }
        let title = t("captions.exportSrt");
        let family = t("captions.subRipSubtitles");
        crate::host::after_dialog(
            move || {
                let chosen = crate::platform::save_file(
                    &title,
                    &folder,
                    &name,
                    (family.as_str(), &["srt"]),
                )?;
                Some(
                    if chosen
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("srt"))
                    {
                        chosen
                    } else {
                        chosen.with_extension("srt")
                    },
                )
            },
            |studio, _, _, path| studio.handle(Msg::Captions(CaptionsMsg::ExportSrtTo(path))),
        );
    }

    /// Writes the captions to `path` as SubRip, and says how many and
    /// where - on a phone, once the phone has put the file where it shows
    /// it.
    fn export_srt(&mut self, studio: &mut Studio, path: &Path) {
        let Some(lane) = export_lane(studio) else {
            self.message = t("captions.noCaptionsToExport");
            return;
        };
        let cues = srt::title_cues(studio.timeline(), Some(&lane.track_id));
        if let Err(error) = std::fs::write(path, srt::write(&cues)) {
            log::warn!("captions: could not write {}: {error}", path.display());
            self.message = tf("captions.couldNotWriteSubtitles", &[&error]);
            return;
        }
        log::info!(
            "captions: wrote {} cue(s) to {}",
            cues.len(),
            path.display()
        );
        self.open = false;
        let count = cues.len();
        // Which lane was written goes without saying when it was the
        // selected title's, or the only lane with titles; otherwise the
        // notice says it was the busiest, and how to pick another.
        let busiest = lane.busiest && srt::title_lanes(studio.timeline()) > 1;
        let say = move |studio: &mut Studio, place: &str| {
            // Each key spelled out where it is used: scripts/locales.py
            // finds the strings the source asks for by reading it.
            let said = if busiest {
                tf("captions.exportedBusiestLane", &[&count, &place])
            } else {
                tf("captions.exportedCaptions", &[&count, &place])
            };
            studio.notify(&said, false);
        };
        let written = path.to_path_buf();
        let handed = crate::platform::publish_document(
            path.to_path_buf(),
            "application/x-subrip",
            move |result| {
                crate::host::on_ui(move |studio, _, _| match result {
                    Ok(place) => say(studio, &place),
                    Err(error) => {
                        log::warn!("captions: could not publish {}: {error}", written.display());
                        studio.notify(
                            &tf(
                                "captions.couldNotPublishSubtitles",
                                &[&written.display(), &error],
                            ),
                            true,
                        );
                    }
                });
            },
        );
        if !handed {
            say(studio, &path.display().to_string());
        }
    }

    /// The sheet opened afresh. `clip` is the clip it was opened on, if
    /// any: the switch starts on when that clip has sound, or when a clip's
    /// menu asked for its voice (`voice`), in which case a clip without
    /// sound falls to the first clip that has some.
    fn open_on(&mut self, studio: &Studio, clip: Option<String>, voice: bool) {
        let installed = installed(&studio.settings.transcribers);
        let model = installed.iter().position(|model| model.active).unwrap_or(0);
        let sources = sources(studio);
        let sounding = clip.filter(|id| sources.iter().any(|source| &source.id == id));
        let from_voice = voice || sounding.is_some();
        let clip = sounding.or_else(|| {
            from_voice
                .then(|| sources.first().map(|source| source.id.clone()))
                .flatten()
        });
        let kept = &studio.prefs.captions;
        *self = CaptionsPane {
            open: true,
            from_voice,
            clip,
            chosen_with: studio.sole_selection(),
            model,
            placement: 0,
            size: 1,
            words: usize::from(kept.words).min(MAX_WORDS),
            animation: usize::from(kept.animation).min(Entrance::ALL.len() - 1),
            upper: kept.upper,
            style: kept
                .style
                .as_ref()
                .and_then(|id| {
                    studio
                        .text_presets
                        .iter()
                        .position(|preset| &preset.id == id)
                })
                .map_or(0, |index| index + 1),
            timings: std::mem::take(&mut self.timings),
            ..CaptionsPane::default()
        };
    }

    /// The layout rows into the preferences, so the next sheet - and the
    /// next launch - opens on them.
    fn remember(&self, studio: &mut Studio) {
        let kept = crate::prefs::CaptionPrefs {
            words: self.words as u8,
            animation: self.animation as u8,
            upper: self.upper,
            style: self
                .style
                .checked_sub(1)
                .and_then(|index| studio.text_presets.get(index))
                .map(|preset| preset.id.clone()),
        };
        if studio.prefs.captions != kept {
            studio.prefs.captions = kept;
            studio.prefs.save(&studio.host.dirs);
        }
    }

    /// The look captions are set in, by the sheet's row: a text preset's,
    /// with the font it brings registered in the same step, or the look
    /// kept by "Use for new titles" (none: the bundled face).
    fn base(&self, studio: &Studio) -> (Option<TextStyle>, Option<Command>) {
        match self
            .style
            .checked_sub(1)
            .and_then(|index| studio.text_presets.get(index))
        {
            Some(preset) => (
                Some(preset.style.clone()),
                crate::presets::install_font(&studio.host.dirs, preset)
                    .map(|(family, path)| Command::AddFont { family, path }),
            ),
            None => (studio.prefs.title_style.clone(), None),
        }
    }

    /// How the words are laid out, by the sheet's rows.
    fn layout(&self) -> Layout {
        Layout {
            words: self.words.min(MAX_WORDS),
            entrance: Entrance::ALL[self.animation.min(Entrance::ALL.len() - 1)],
            upper: self.upper,
        }
    }

    /// The clip the panel transcribes, and the one its list shows - the
    /// same answer for both, so it never runs a clip it is not showing.
    /// The editor stays live under the panel: a clip with sound selected
    /// since the panel's clip was chosen is the one meant now, and is
    /// taken. Otherwise the chosen clip while it is still on the timeline
    /// with sound, and none - nothing to run - when it is not.
    fn subject_clip(&self, studio: &Studio, sources: &[Source]) -> Option<String> {
        follow_selection(
            studio.sole_selection(),
            self.chosen_with.as_deref(),
            self.clip.as_deref(),
            |id| sources.iter().any(|source| source.id == id),
        )
    }

    /// A caption's look, by the sheet's rows: where it sits and its size.
    fn look(&self) -> (f64, f64) {
        (
            CAPTION_OFFSETS[self.placement.min(2)],
            CAPTION_SIZES[self.size.min(2)],
        )
    }

    /// The script as titles, one after another from the playhead.
    fn run_script(&mut self, studio: &mut Studio) {
        let lines = script_captions(&self.text);
        if lines.is_empty() {
            self.message = t("captions.nothingToCaptionYet");
            return;
        }
        let look = self.look();
        let layout = self.layout();
        let mut at = f64::from(studio.playhead);
        let (base, font) = self.base(studio);
        // Each line's span from the playhead on, then cut into a few words
        // at a time when the sheet asks for that, each held by its length.
        let mut spans = Vec::new();
        for (text, seconds) in lines {
            if layout.words == 0 {
                spans.push((text, at, at + seconds));
            } else {
                let segment = Segment {
                    start: at,
                    end: at + seconds,
                    text,
                    words: Vec::new(),
                };
                spans.extend(word_captions(&[segment], layout.words));
            }
            at += seconds;
        }
        let mut commands = laid_out(spans, |seconds| seconds, look, layout, base.as_ref());
        let count = commands.len();
        if let Some(font) = font {
            commands.insert(0, font);
        }
        self.open = false;
        studio.apply(Command::Batch { commands });
        studio.notify(&tf("captions.addedCaptions", &[&count]), false);
    }

    /// The sheet's clip through the transcriber on a worker, reporting into
    /// the sheet as it goes.
    fn run_sound(&mut self, studio: &mut Studio) {
        // The clip the panel shows, and no other: see `subject_clip`.
        let Some(clip) = self
            .subject_clip(studio, &sources(studio))
            .and_then(|id| studio.clip(&id))
            .cloned()
        else {
            self.message = t("captions.clipNoLongerTimeline");
            return;
        };
        let Some(media) = studio.project().media_by_id(&clip.media_id).cloned() else {
            self.message = t("captions.clipHasNoFile");
            return;
        };
        let Some((model, megabytes)) = installed(&studio.settings.transcribers)
            .get(self.model)
            .map(|model| (model.id.clone(), model.megabytes))
        else {
            self.message = t("captions.downloadTranscriberModelFirst");
            return;
        };
        let request = concat_speech::transcribe::TranscribeRequest {
            path: media.path.clone(),
            audio_stream: clip.audio_stream,
            source_start: clip.source_start,
            window: clip.duration * clip.speed,
            model_id: model.clone(),
        };
        let dirs = studio.host.dirs.clone();
        let transcriber = Arc::clone(&studio.host.transcriber);
        // The run as the bar lays it out: the sound read in at a few
        // hundred times its length, the model loaded unless it still is,
        // and whisper a window at a time - at the pace the last run on
        // this model measured, or the estimate by its size.
        let windows = (request.window / 30.0).ceil().max(1.0) as f32;
        let learned = self.timings.get(&model);
        let loaded = studio.host.transcriber.loaded_model().as_deref() == Some(model.as_str());
        let load = if loaded {
            0.0
        } else {
            learned.load.unwrap_or(0.5 + megabytes / 400.0)
        };
        let per_window = learned
            .per_unit
            .unwrap_or(0.3 + megabytes * WINDOW_SECONDS_PER_MB);
        self.pace = Pace::new(vec![
            (Stage::Reading, 0.2 + request.window as f32 / 400.0),
            (Stage::Loading, load),
            (Stage::Working, windows * per_window),
        ]);
        let (base, font) = self.base(studio);
        self.subject = Some(Subject {
            start: clip.start,
            speed: clip.speed,
            look: self.look(),
            layout: self.layout(),
            base,
            font,
            model,
            windows,
        });
        self.running = true;
        self.message.clear();
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                transcriber.transcribe(&dirs, &request, move |stage| {
                    on_ui_in_project(epoch, move |studio, _, _| {
                        studio.handle(Msg::Captions(CaptionsMsg::Progress(stage)));
                    });
                })
            },
            |studio, _, _, result| studio.handle(Msg::Captions(CaptionsMsg::Finished(result))),
        );
    }

    /// The sheet as Slint shows it.
    pub fn data(&self, studio: &Studio) -> CaptionsSheetData {
        let sources = if self.open {
            sources(studio)
        } else {
            Vec::new()
        };
        let source = self
            .subject_clip(studio, &sources)
            .and_then(|id| sources.iter().position(|source| source.id == id))
            .map_or(-1, |index| index as i32);
        let labels: Vec<slint::SharedString> = sources
            .into_iter()
            .map(|source| source.label.into())
            .collect();
        let styles: Vec<slint::SharedString> = if self.open {
            std::iter::once(t("captions.keptStyle"))
                .chain(studio.text_presets.iter().map(|preset| preset.name.clone()))
                .map(Into::into)
                .collect()
        } else {
            Vec::new()
        };
        CaptionsSheetData {
            open: self.open,
            from_voice: self.from_voice,
            sources: std::rc::Rc::new(slint::VecModel::from(labels)).into(),
            source,
            text: self.text.as_str().into(),
            model: self.model as i32,
            placement: self.placement as i32,
            size: self.size as i32,
            words: self.words as i32,
            animation: self.animation as i32,
            upper: self.upper,
            styles: std::rc::Rc::new(slint::VecModel::from(styles)).into(),
            style: self.style as i32,
            running: self.running,
            progress: self.pace.floor,
            ceiling: self.pace.ceiling,
            step: self.pace.step,
            left: self.pace.left,
            stage: match self.pace.stage() {
                Some(Stage::Loading) => t("captions.loadingTheModel"),
                Some(Stage::Working) => t("captions.transcribing"),
                _ => t("captions.readingTheSound"),
            }
            .into(),
            ready: !installed(&studio.settings.transcribers).is_empty(),
            message: self.message.as_str().into(),
        }
    }
}

/// Imported cues keep the file's timing, including captions shorter than
/// generated speech segments. The title command applies the engine's minimum
/// clip duration where needed.
fn srt_commands(cues: Vec<srt::Cue>, look: (f64, f64), base: Option<&TextStyle>) -> Vec<Command> {
    cues.into_iter()
        .filter(|cue| !cue.text.trim().is_empty())
        .map(|cue| {
            caption_clip(
                cue.text.trim().to_owned(),
                cue.start,
                cue.end - cue.start,
                look,
                base,
                Vec::new(),
            )
        })
        .collect()
}

/// The lane an SRT export writes out.
struct ExportLane {
    track_id: String,
    /// Chosen as the lane with the most titles, not by a selected title.
    busiest: bool,
}

/// The lane whose titles are written out as captions: the selected
/// title's; or with no title selected, the lane with the most titles -
/// captions come many to a lane, a title card one or two, and nothing
/// else in the document tells them apart (see `srt::busiest_title_lane`).
/// None when the timeline has no title.
fn export_lane(studio: &Studio) -> Option<ExportLane> {
    let selected = studio
        .sole_selection()
        .and_then(|id| studio.clip(&id))
        .filter(|clip| clip.kind == ClipKind::Text)
        .map(|clip| clip.track_id.clone());
    match selected {
        Some(track_id) => Some(ExportLane {
            track_id,
            busiest: false,
        }),
        None => srt::busiest_title_lane(studio.timeline()).map(|track_id| ExportLane {
            track_id,
            busiest: true,
        }),
    }
}

/// The name an export is offered under: the project's, as one file name,
/// or "captions" with no project or no usable name.
fn file_stem(studio: &Studio) -> String {
    let name = studio
        .session
        .as_ref()
        .map(|session| session.settings().name)
        .unwrap_or_default();
    let stem: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let stem = stem.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if stem.is_empty() {
        "captions".to_owned()
    } else {
        stem.to_owned()
    }
}

/// `name` in `dir`, or `name (2)`, `name (3)`... until nothing there is
/// written over.
fn unique_path(dir: &Path, name: &str) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let (stem, ext) = match name.rfind('.') {
        Some(dot) if dot > 0 => (&name[..dot], &name[dot..]),
        _ => (name, ""),
    };
    (2..)
        .map(|n| dir.join(format!("{stem} ({n}){ext}")))
        .find(|path| !path.exists())
        .unwrap_or(first)
}

/// [`CaptionsPane::subject_clip`] with the studio taken out: `selected`
/// the sole selection, `chosen_with` the selection when `clip` was chosen,
/// and `sounds` whether a clip is on the timeline with sound.
fn follow_selection(
    selected: Option<String>,
    chosen_with: Option<&str>,
    clip: Option<&str>,
    sounds: impl Fn(&str) -> bool,
) -> Option<String> {
    if let Some(selected) = selected.filter(|id| sounds(id))
        && Some(selected.as_str()) != chosen_with
    {
        return Some(selected);
    }
    clip.filter(|id| sounds(id)).map(str::to_owned)
}

/// A clip on the timeline whose voice can be captioned, as the sheet's
/// list shows it.
struct Source {
    id: String,
    label: String,
}

/// The clips on the current timeline that have sound, in the order they
/// play: an audio clip, or a video whose file has an audio stream.
fn sources(studio: &Studio) -> Vec<Source> {
    let project = studio.project();
    let mut clips: Vec<_> = studio
        .timeline()
        .clips
        .iter()
        .filter(|clip| studio.clip_has_sound(clip))
        .map(|clip| {
            let name = project
                .media_by_id(&clip.media_id)
                .map_or(clip.name.as_str(), |media| media.name.as_str());
            (clip.start, clip.id.clone(), name.to_owned())
        })
        .collect();
    clips.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    clips
        .into_iter()
        .map(|(start, id, name)| Source {
            id,
            label: source_label(&name, start),
        })
        .collect()
}

/// A clip as the list names it: its file and where it starts on the
/// timeline, so two clips of one file are told apart.
fn source_label(name: &str, start: f64) -> String {
    let seconds = start.max(0.0).floor() as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{name} · {hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{name} · {minutes}:{seconds:02}")
    }
}

/// How a run lays its words out: how many at a time, how each caption
/// comes in, and whether in capitals. Chosen in the sheet's rows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// 0 a phrase as spoken; else words at a time.
    words: usize,
    entrance: Entrance,
    upper: bool,
}

/// How a caption comes in: keys at the head of its title, so it can be
/// changed afterwards like any other key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entrance {
    None,
    /// Grows past its size and settles: the short-video caption.
    Pop,
    /// Fades up from nothing.
    Fade,
    /// Rises into place as it fades up.
    Rise,
}

impl Entrance {
    /// In the sheet's order.
    const ALL: [Entrance; 4] = [
        Entrance::None,
        Entrance::Pop,
        Entrance::Fade,
        Entrance::Rise,
    ];

    /// The keys for a caption `duration` seconds long sitting at
    /// `offset_y`. The entrance takes a fixed time - a fifth of a second
    /// or less - and never more than half of a short caption, so the
    /// words are still for most of the time they are up.
    fn keys(self, duration: f64, offset_y: f64) -> Vec<ClipKey> {
        if duration <= 0.0 || self == Entrance::None {
            return Vec::new();
        }
        let at = |seconds: f64| (seconds / duration).clamp(0.0, 1.0);
        let key = |property, seconds: f64, value, ease| ClipKey {
            property,
            at: at(seconds),
            value,
            ease,
        };
        match self {
            Entrance::None => Vec::new(),
            Entrance::Pop => {
                let length = 0.16_f64.min(duration * 0.5);
                vec![
                    key(KeyProperty::Scale, 0.0, 0.6, KeyEase::LINEAR),
                    key(KeyProperty::Scale, length * 0.65, 1.1, KeyEase::OUT),
                    key(KeyProperty::Scale, length, 1.0, KeyEase::IN_OUT),
                    key(KeyProperty::Opacity, 0.0, 0.0, KeyEase::LINEAR),
                    key(KeyProperty::Opacity, length * 0.4, 1.0, KeyEase::OUT),
                ]
            }
            Entrance::Fade => {
                let length = 0.2_f64.min(duration * 0.5);
                vec![
                    key(KeyProperty::Opacity, 0.0, 0.0, KeyEase::LINEAR),
                    key(KeyProperty::Opacity, length, 1.0, KeyEase::OUT),
                ]
            }
            Entrance::Rise => {
                let length = 0.2_f64.min(duration * 0.5);
                vec![
                    key(KeyProperty::OffsetY, 0.0, offset_y + 0.03, KeyEase::LINEAR),
                    key(KeyProperty::OffsetY, length, offset_y, KeyEase::OUT),
                    key(KeyProperty::Opacity, 0.0, 0.0, KeyEase::LINEAR),
                    key(KeyProperty::Opacity, length, 1.0, KeyEase::OUT),
                ]
            }
        }
    }
}

/// A finished transcription as captions on the timeline: a phrase at a
/// time as whisper heard it, or a few words at a time by their own
/// timing, with the entrance and capitals the run was started with.
fn spoken_captions(
    segments: &[Segment],
    subject: &Subject,
    base: Option<&TextStyle>,
) -> Vec<Command> {
    let spans = if subject.layout.words == 0 {
        segments
            .iter()
            .map(|segment| (segment.text.trim().to_owned(), segment.start, segment.end))
            .collect()
    } else {
        word_captions(segments, subject.layout.words)
    };
    let (start, speed) = (subject.start, subject.speed);
    laid_out(
        spans,
        |seconds| start + seconds / speed,
        subject.look,
        subject.layout,
        base,
    )
}

/// Captions as title commands: each span's words, with `timeline` taking
/// its seconds onto the timeline. A phrase is held a fifth of a second at
/// least, as it always was. Words at a time are laid end to end - each
/// held to the next one's start where they touch - and a hair short of
/// it, so the next lands on the same lane rather than on one above.
fn laid_out(
    spans: Vec<(String, f64, f64)>,
    timeline: impl Fn(f64) -> f64,
    look: (f64, f64),
    layout: Layout,
    base: Option<&TextStyle>,
) -> Vec<Command> {
    let spans: Vec<(String, f64, f64)> = spans
        .into_iter()
        .map(|(text, start, end)| (text.trim().to_owned(), timeline(start), timeline(end)))
        .filter(|(text, _, _)| !text.is_empty())
        .collect();
    let starts: Vec<f64> = spans.iter().map(|span| span.1).collect();
    spans
        .into_iter()
        .enumerate()
        .map(|(index, (text, start, end))| {
            let duration = if layout.words == 0 {
                (end - start).max(0.2)
            } else {
                let length = (end - start).max(1.0 / 30.0);
                match starts.get(index + 1) {
                    Some(&next) if start + length >= next - 1e-6 => {
                        (next - start - 1e-6).max(1.0 / 60.0)
                    }
                    _ => length,
                }
            };
            let text = if layout.upper {
                text.to_uppercase()
            } else {
                text
            };
            let keys = layout.entrance.keys(duration, look.0);
            caption_clip(text, start, duration, look, base, keys)
        })
        .collect()
}

/// Captions of a few words at a time, in the transcription's own seconds
/// as `(words, start, end)`: `per` words each, fewer where a sentence or
/// a clause ends, where the speaker pauses, or where whisper's phrase
/// does. A word's timing is whisper's when it gave one, and its share of
/// the phrase otherwise. A short gap between two captions is closed, and
/// a caption shorter than [`MIN_WORDS_SECONDS`] is held longer where the
/// next leaves room. A full stop or comma at a caption's end is dropped -
/// three words on screen do not need it - while a question or an
/// exclamation keeps its mark.
fn word_captions(segments: &[Segment], per: usize) -> Vec<(String, f64, f64)> {
    let per = per.max(1);
    let mut chunks: Vec<(String, f64, f64)> = Vec::new();
    for segment in segments {
        let words = segment.timed_words();
        let mut run: Vec<&concat_speech::transcribe::Word> = Vec::new();
        for (index, word) in words.iter().enumerate() {
            run.push(word);
            let next = words.get(index + 1);
            let pause = next.is_some_and(|next| next.start - word.end > PAUSE_SECONDS);
            let clause = word
                .text
                .ends_with(['.', ',', '!', '?', ';', ':', '\u{2026}']);
            if run.len() >= per || pause || clause || next.is_none() {
                let text: Vec<&str> = run.iter().map(|word| word.text.as_str()).collect();
                let text = text
                    .join(" ")
                    .trim_end_matches(['.', ',', ';', ':'])
                    .to_owned();
                let start = run[0].start;
                let end = run[run.len() - 1].end.max(start);
                if !text.is_empty() {
                    chunks.push((text, start, end));
                }
                run.clear();
            }
        }
    }
    for index in 0..chunks.len() {
        let start = chunks[index].1;
        let next = chunks.get(index + 1).map(|chunk| chunk.1.max(start));
        let mut end = chunks[index].2;
        if let Some(next) = next
            && next - end < HOLD_SECONDS
        {
            end = next;
        }
        if end - start < MIN_WORDS_SECONDS {
            end = next.map_or(start + MIN_WORDS_SECONDS, |next| {
                (start + MIN_WORDS_SECONDS).min(next)
            });
        }
        if let Some(next) = next {
            end = end.min(next);
        }
        chunks[index].2 = end.max(start);
    }
    chunks
}

/// One caption as a title clip: its words, when and for how long, its
/// look, and the keys it comes in with.
fn caption_clip(
    text: String,
    start: f64,
    duration: f64,
    look: (f64, f64),
    base: Option<&TextStyle>,
    keys: Vec<ClipKey>,
) -> Command {
    let (offset_y, font_size) = look;
    // The kept look where there is one - family, weight, colours, stroke,
    // shadow, background - at the caption's own size and place; the
    // bundled face otherwise.
    let style = match base {
        Some(base) => TextStyle {
            content: text,
            font_size,
            max_width: 0.0,
            max_height: 0.0,
            ..base.clone()
        },
        None => TextStyle {
            content: text,
            font_family: "Hanken Grotesk".to_owned(),
            font_size,
            font_weight: 600.0,
            ..TextStyle::default()
        },
    };
    Command::AddTextClip {
        track_id: None,
        above: true,
        start,
        style: Some(style),
        duration: Some(duration),
        offset_y: Some(offset_y),
        keys,
    }
}

/// Longest a caption line gets before it is wrapped: about what two lines
/// of broadcast subtitle hold, and what a reader takes in at a glance.
const CAPTION_CHARS: usize = 42;

/// A script as caption lines, each with how long it stays up: a line's
/// reading time at [`CHARS_PER_SECOND`], held to one second at least so
/// a short word is not a flicker, and seven at most so a long line does
/// not hang. A line break in the script is a break the author asked for;
/// within a paragraph a sentence is a caption, and a long sentence wraps
/// at its words.
fn script_captions(text: &str) -> Vec<(String, f64)> {
    text.lines()
        .flat_map(sentences)
        .flat_map(|sentence| wrap_caption(&sentence))
        .map(|line| {
            let seconds =
                (line.chars().count() as f64 / f64::from(CHARS_PER_SECOND)).clamp(1.0, 7.0);
            (line, seconds)
        })
        .collect()
}

/// A paragraph's sentences. A full stop, question or exclamation mark ends
/// one when it is followed by space or by the end - so "3.5" and "e.g." hold
/// together - and the CJK marks end one on their own.
fn sentences(paragraph: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(ch) = chars.next() {
        current.push(ch);
        let ends = match ch {
            '。' | '！' | '？' => true,
            '.' | '!' | '?' => chars.peek().is_none_or(|next| next.is_whitespace()),
            _ => false,
        };
        if ends {
            let sentence = current.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_owned());
            }
            current.clear();
        }
    }
    let rest = current.trim();
    if !rest.is_empty() {
        out.push(rest.to_owned());
    }
    out
}

/// A sentence in lines of at most [`CAPTION_CHARS`], broken between words;
/// a word longer than a line, or a run of CJK with no spaces, is broken
/// where it must be.
fn wrap_caption(sentence: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_chars = 0;
    for word in sentence.split_whitespace() {
        let word_chars = word.chars().count();
        if line_chars > 0 && line_chars + 1 + word_chars > CAPTION_CHARS {
            lines.push(std::mem::take(&mut line));
            line_chars = 0;
        }
        if word_chars > CAPTION_CHARS {
            let mut piece = String::new();
            for ch in word.chars() {
                piece.push(ch);
                if piece.chars().count() == CAPTION_CHARS {
                    lines.push(std::mem::take(&mut piece));
                }
            }
            line = piece;
            line_chars = line.chars().count();
            continue;
        }
        if line_chars > 0 {
            line.push(' ');
            line_chars += 1;
        }
        line.push_str(word);
        line_chars += word_chars;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{script_captions, source_label};
    use concat_project::model::KeyProperty;
    use concat_speech::transcribe::{Segment, Word};

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            start,
            end,
            text: text.to_owned(),
        }
    }

    fn spoken() -> Vec<Segment> {
        vec![Segment {
            start: 0.0,
            end: 2.8,
            text: "Olá, tudo bem com você? Sim.".to_owned(),
            words: vec![
                word("Olá,", 0.0, 0.3),
                word("tudo", 0.4, 0.6),
                word("bem", 0.6, 0.8),
                word("com", 0.85, 1.0),
                word("você?", 1.0, 1.3),
                word("Sim.", 2.5, 2.8),
            ],
        }]
    }

    /// Three at a time, fewer at a comma, a question and a pause; the full
    /// stop and the comma go, the question mark stays; a short gap closes
    /// and a long one does not.
    #[test]
    fn words_come_a_few_at_a_time_by_their_own_timing() {
        let chunks = super::word_captions(&spoken(), 3);
        let text: Vec<&str> = chunks.iter().map(|chunk| chunk.0.as_str()).collect();
        assert_eq!(text, ["Olá", "tudo bem com", "você?", "Sim"]);
        assert_eq!(chunks[0].1, 0.0);
        assert!((chunks[0].2 - 0.4).abs() < 1e-9, "the short gap is closed");
        assert!((chunks[2].2 - 1.3).abs() < 1e-9, "the pause is kept");
        assert!((chunks[3].1 - 2.5).abs() < 1e-9);
        for pair in chunks.windows(2) {
            assert!(pair[0].2 <= pair[1].1 + 1e-9);
        }
        let one = super::word_captions(&spoken(), 1);
        assert_eq!(one.len(), 6);
    }

    /// A segment whisper gave no word times for is spread over its span.
    #[test]
    fn a_phrase_without_word_times_is_still_cut_into_words() {
        let segment = Segment {
            start: 1.0,
            end: 3.0,
            text: "um dois três quatro".to_owned(),
            words: Vec::new(),
        };
        let chunks = super::word_captions(&[segment], 2);
        let text: Vec<&str> = chunks.iter().map(|chunk| chunk.0.as_str()).collect();
        assert_eq!(text, ["um dois", "três quatro"]);
        assert_eq!(chunks[0].1, 1.0);
        assert!((chunks[1].2 - 3.0).abs() < 1e-9);
    }

    /// Words at a time land end to end on one lane - not stepped up a lane
    /// each where two touch - in capitals, each with its entrance keyed.
    #[test]
    fn word_captions_share_a_lane_and_come_in_keyed() {
        let layout = super::Layout {
            words: 2,
            entrance: super::Entrance::Pop,
            upper: true,
        };
        let spans = super::word_captions(&spoken(), 2);
        let count = spans.len();
        let commands = super::laid_out(
            spans,
            |seconds| 10.0 + seconds / 1.5,
            (0.35, 0.05),
            layout,
            None,
        );
        let mut editor = concat_project::Editor::new();
        editor
            .apply(concat_project::Command::Batch { commands })
            .unwrap();
        let clips = &editor.project().active().clips;
        assert_eq!(clips.len(), count);
        assert!(clips.iter().all(|clip| clip.track_id == clips[0].track_id));
        let first = clips
            .iter()
            .min_by(|a, b| a.start.total_cmp(&b.start))
            .expect("a caption");
        assert_eq!(
            first.text.as_ref().map(|style| style.content.as_str()),
            Some("OLÁ")
        );
        assert!((first.start - 10.0).abs() < 1e-9);
        for clip in clips.iter() {
            let scale: Vec<_> = clip
                .keys
                .iter()
                .filter(|key| key.property == KeyProperty::Scale)
                .collect();
            assert_eq!(scale.len(), 3);
            assert!((scale.last().expect("a key").value - 1.0).abs() < 1e-9);
            assert!(
                clip.keys
                    .iter()
                    .all(|key| (0.0..=0.5 + 1e-9).contains(&key.at))
            );
        }
        // One undo takes the whole run away.
        assert!(editor.undo());
        assert!(editor.project().active().clips.is_empty());
    }

    #[test]
    fn each_entrance_settles_where_the_caption_sits() {
        assert!(super::Entrance::None.keys(1.0, 0.35).is_empty());
        for entrance in [
            super::Entrance::Pop,
            super::Entrance::Fade,
            super::Entrance::Rise,
        ] {
            for duration in [0.1, 0.5, 3.0] {
                let keys = entrance.keys(duration, 0.35);
                assert!(!keys.is_empty());
                assert!(keys.iter().all(|key| (0.0..=1.0).contains(&key.at)));
                let last = |property| {
                    keys.iter()
                        .filter(|key| key.property == property)
                        .max_by(|a, b| a.at.total_cmp(&b.at))
                        .map(|key| key.value)
                };
                assert_eq!(last(KeyProperty::Opacity), Some(1.0));
                if entrance == super::Entrance::Rise {
                    assert_eq!(last(KeyProperty::OffsetY), Some(0.35));
                }
            }
        }
    }

    #[test]
    fn imported_srt_preserves_short_cues_and_multiline_text() {
        let cues = concat_project::srt::parse(
            "1\n00:00:01,000 --> 00:00:01,100\nOlá\nAção\n\n2\n00:00:02,000 --> 00:00:03,000\nFim\n",
        ).unwrap();
        let commands = super::srt_commands(cues.clone(), (0.35, 0.04), None);
        let mut editor = concat_project::Editor::new();
        editor
            .apply(concat_project::Command::Batch { commands })
            .unwrap();
        let exported = concat_project::srt::write(&concat_project::srt::title_cues(
            editor.project().active(),
            None,
        ));
        assert_eq!(concat_project::srt::parse(&exported).unwrap(), cues);
        assert!(editor.undo());
        assert!(editor.project().active().clips.is_empty());
        assert!(editor.redo());
        assert_eq!(
            concat_project::srt::title_cues(editor.project().active(), None).len(),
            2
        );
    }

    /// The panel floats over a live editor: a clip with sound selected
    /// since the panel's clip was chosen is the one it shows and runs; a
    /// pick from its list holds until the selection moves again; and a
    /// chosen clip gone from the timeline is not run unseen.
    #[test]
    fn the_captioned_clip_follows_a_new_selection_of_a_clip_with_sound() {
        let sounds = |id: &str| ["a", "b", "c"].contains(&id);
        let follow = |selected: Option<&str>, chosen_with: Option<&str>, clip: Option<&str>| {
            super::follow_selection(selected.map(str::to_owned), chosen_with, clip, sounds)
        };
        // Opened on A, A still selected: A.
        assert_eq!(
            follow(Some("a"), Some("a"), Some("a")).as_deref(),
            Some("a")
        );
        // B selected on the timeline since: B.
        assert_eq!(
            follow(Some("b"), Some("a"), Some("a")).as_deref(),
            Some("b")
        );
        // C picked from the list with B selected: C, until the selection moves.
        assert_eq!(
            follow(Some("b"), Some("b"), Some("c")).as_deref(),
            Some("c")
        );
        // A title selected - no sound - leaves the choice alone.
        assert_eq!(
            follow(Some("t"), Some("b"), Some("c")).as_deref(),
            Some("c")
        );
        // The chosen clip deleted: nothing to run, rather than a clip unseen.
        assert_eq!(follow(None, None, Some("gone")), None);
    }

    #[test]
    fn imported_srt_ignores_empty_captions() {
        let cues = vec![concat_project::srt::Cue {
            start: 0.0,
            end: 1.0,
            text: "  ".into(),
        }];
        assert!(super::srt_commands(cues, (0.35, 0.04), None).is_empty());
    }

    /// A clip in the voice list is its file and where it starts, in
    /// minutes and seconds, with hours once there are any.
    #[test]
    fn a_voice_is_named_by_its_file_and_where_it_starts() {
        assert_eq!(source_label("Interview.mp4", 0.0), "Interview.mp4 · 0:00");
        assert_eq!(source_label("Interview.mp4", 72.9), "Interview.mp4 · 1:12");
        assert_eq!(source_label("Talk.wav", 3_725.0), "Talk.wav · 1:02:05");
        assert_eq!(source_label("Talk.wav", -1.0), "Talk.wav · 0:00");
    }

    /// A script becomes one caption per sentence, a hand line break is
    /// kept, a long sentence wraps at its words, and each line is held for
    /// its reading time within one to seven seconds.
    #[test]
    fn a_script_is_cut_into_readable_lines() {
        let lines = script_captions(
            "Hello there. This is version 3.5, mind!\n\nA sentence that runs on for far \
             longer than a caption line has any business running on for. Ok?",
        );
        let text: Vec<&str> = lines.iter().map(|(line, _)| line.as_str()).collect();
        assert_eq!(
            text,
            [
                "Hello there.",
                "This is version 3.5, mind!",
                "A sentence that runs on for far longer",
                "than a caption line has any business",
                "running on for.",
                "Ok?",
            ]
        );
        assert!(
            lines
                .iter()
                .all(|(_, seconds)| (1.0..=7.0).contains(seconds))
        );
        assert_eq!(lines[0].1, 1.0);
        assert!(lines[2].1 > lines[0].1);
        assert!(script_captions("  \n ").is_empty());
        assert_eq!(script_captions("你好。再见！").len(), 2);
    }
}
