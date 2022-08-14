use similari::examples::BoxGen2;
use similari::prelude::PositionalMetricType::IoU;
use similari::prelude::Sort;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::utils::bbox::BoundingBox;

fn main() {
    let mut tracker = Sort::new(1, 10, 1, IoU(DEFAULT_SORT_IOU_THRESHOLD), None);

    let pos_drift = 1.0;
    let box_drift = 0.2;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = (b1.next().unwrap().into(), Some(1));
        let obj2b = (b2.next().unwrap().into(), Some(2));
        let _tracks = tracker.predict(&[obj1b, obj2b]);
        eprintln!("Tracks: {:#?}", _tracks);
    }

    tracker.skip_epochs(2);

    let tracks = tracker.wasted();
    for t in tracks {
        eprintln!("Track id: {}", t.get_track_id());
        eprintln!(
            "Boxes: {:#?}",
            t.get_attributes()
                .predicted_boxes
                .iter()
                .map(BoundingBox::from)
                .collect::<Vec<_>>()
        );
    }
}
