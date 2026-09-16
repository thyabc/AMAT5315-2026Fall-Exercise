use clap::Parser;
use rand::RngExt;
use std::fs::{File, create_dir_all};
use std::io::Write;

#[derive(Parser, Debug)]
struct Args {

    #[arg(long)]
    update:String,

    #[arg(long)]
    l:usize,

    #[arg(long)]
    t_from:f64,

    #[arg(long)]
    t_to:f64,

    #[arg(long)]
    t_step:f64,

    #[arg(long)]
    discard:usize,

    #[arg(long)]
    measure:usize,

    #[arg(long, default_value_t=0)]
every:usize,

    #[arg(long)]
    seed:u64,

    #[arg(long)]
    out:String,
}


// Ising模型结构
struct IsingModel {

    // 晶格大小
    l:usize,

    // spin状态
    // 每个格点 +1 或 -1
    spins:Vec<i8>,

    // 温度
    temperature:f64,
}


// 初始化模型
impl IsingModel {

    fn new(l:usize, temperature:f64)->Self {

        Self{

            l,

            // 初始全部向上
            spins:vec![1;l*l],

            temperature,

        }
    }
fn metropolis_step(&mut self){


    let mut rng = rand::rng();


    // 随机选择一个格点
    let index =
        rng.random_range(0..self.l*self.l);


    // 当前翻转能量
    let de =
        self.delta_energy(index);



    // 接受概率
    let accept =
        if de <=0.0 {

            true

        }else{

            let r:f64=rng.random();

            r < (-de/self.temperature).exp()

        };


    if accept{

        self.flip(index);

    }

}
fn sweep(&mut self){

    let n = self.l*self.l;


    for _ in 0..n {

        self.metropolis_step();

    }

}
    fn delta_energy(&self,index:usize)->f64{


    let row=index/self.l;
    let col=index%self.l;


    let up=((row+self.l-1)%self.l)*self.l+col;

    let down=((row+1)%self.l)*self.l+col;

    let left=row*self.l+(col+self.l-1)%self.l;

    let right=row*self.l+(col+1)%self.l;


    let neighbour_sum =
        self.spins[up]
        + self.spins[down]
        + self.spins[left]
        + self.spins[right];


    2.0*
    self.spins[index] as f64
    *
    neighbour_sum as f64
}
fn flip(&mut self,index:usize){

    self.spins[index]*=-1;

}
    fn magnetization(&self)->f64{

        let sum:i32=self.spins
            .iter()
            .map(|&x| x as i32)
            .sum();


        sum as f64/(self.l*self.l) as f64
    }
    // 计算总能量
    fn energy(&self)->f64{

        let mut e=0.0;


        for i in 0..self.l{

            for j in 0..self.l{


                let idx=i*self.l+j;


                let spin=self.spins[idx] as f64;


                // 周期边界
                let right=
                    i*self.l+(j+1)%self.l;


                let down=
                    ((i+1)%self.l)*self.l+j;


                e -= spin*(self.spins[right] as f64);

                e -= spin*(self.spins[down] as f64);


            }

        }


        e
    }
}



fn main(){

    let args=Args::parse();
create_dir_all(&args.out).unwrap();
let mut run_file =
    File::create(
        format!("{}/run.json",&args.out)
    ).unwrap();
let mut series_file =
    File::create(
        format!("{}/series.jsonl", args.out)
    ).unwrap();
let mut spins_file =
    File::create(
        format!("{}/spins.jsonl", args.out)
    ).unwrap();
writeln!(
    run_file,
    "{{\"L\":{},\"update\":\"{}\",\"t_from\":{},\"t_to\":{},\"t_step\":{},\"discard\":{},\"measure\":{},\"seed\":{}}}",
    args.l,
    args.update,
    args.t_from,
    args.t_to,
    args.t_step,
    args.discard,
    args.measure,
    args.seed
).unwrap();

    println!("Ising simulation start");

    println!("update = {}",args.update);
    println!("L = {}",args.l);
    println!(
        "temperature {} -> {}",
        args.t_from,
        args.t_to
    );

    println!("output = {}",args.out);



    // 创建模型
    let mut model=
        IsingModel::new(
            args.l,
            args.t_from
        );

   let mut t=args.t_from;


while t <= args.t_to {


    println!("Temperature = {}",t);



    model.temperature=t;



    // equilibration
    for sweep in 0..args.discard {

        model.sweep();

    }


    // measurement
    for sweep in 0..args.measure {

    model.sweep();


    let m = model.magnetization();

    let e = model.energy();


    println!(
        "M={} E={}",
        m,
        e
    );


    writeln!(
        series_file,
        "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{},\"E\":{}}}",
        args.l,
        model.temperature,
        sweep,
        m,
        e
    ).unwrap();


if args.every > 0 && sweep % args.every == 0 {


    writeln!(
        spins_file,
        "{{\"L\":{},\"T\":{},\"sweep\":{},\"spins\":{:?}}}",
        args.l,
        model.temperature,
        sweep,
        model.spins
    ).unwrap();


}
}

    t += args.t_step;


}
    println!(
        "Initial magnetization = {}",
        model.magnetization()
    );
    println!(
    "Initial energy = {}",
    model.energy()
);
let idx=0;

println!(
"Delta E = {}",
model.delta_energy(idx)
);
model.metropolis_step();


println!(
"After one step magnetization = {}",
model.magnetization()
);
}
