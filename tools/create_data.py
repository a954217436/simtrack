
from pathlib import Path
import fire
from det3d.datasets.utils.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(root_path, version, nsweeps=10):
    
    import det3d.datasets.nuscenes.nuscenes_tracking as nu_ds
    nu_ds.create_nuscenes_tracking_infos(root_path, version=version, nsweeps=nsweeps)
    
    
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_train_{:02d}sweeps_tracking.pkl".format(nsweeps),
        nsweeps=1,
    )


def waymo_data_prep(root_path, split, nsweeps=1):
    # from det3d.datasets.waymo import waymo_common as waymo_ds
    from det3d.datasets.waymo import waymo_tracking as waymo_ds

    waymo_ds.create_waymo_tracking_infos(root_path, split=split, nsweeps=nsweeps)
    # waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "simtrack_infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )

if __name__ == "__main__":
    fire.Fire()
