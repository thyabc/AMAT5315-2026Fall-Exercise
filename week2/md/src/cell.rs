#[derive(Clone, Debug)]
pub struct CellList {
    pub cells: Vec<Vec<usize>>,
    pub ncell: [usize; 2],
    pub cell_size: f64,
}

impl CellList {
    pub fn new(ncell: [usize;2], cell_size:f64) -> Self {
        let n = ncell[0]*ncell[1];
        Self {
            cells: vec![Vec::new(); n],
            ncell,
            cell_size,
        }
    }

    fn index(&self, ix:usize, iy:usize)->usize {
        iy*self.ncell[0]+ix
    }

    pub fn clear(&mut self){
        for c in self.cells.iter_mut(){
            c.clear();
        }
    }
}
