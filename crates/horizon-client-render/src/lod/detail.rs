//! Far detail.
use super::*;

impl Lod {
    /// Far detail (`graphics::FarDetail`): the next update replans when it
    /// changes, tiles are sampled again, and with Reduced tiles sampled from
    /// then on approximate wide cells (`GeneratorWorker::lod_grid`).
    pub fn set_far_detail(&mut self, detail: FarDetail) {
        if self.far_detail != detail {
            let keys: Vec<_> = self
                .ready
                .keys()
                .chain(self.pending.iter())
                .copied()
                .collect();
            for key in keys {
                self.mark_stale(key);
            }
        }
        self.far_detail = detail;
        let reduced = detail == FarDetail::Reduced;
        self.shared.approximate.store(reduced, Ordering::Relaxed);
    }
}
