"""
Test the global distributed info
"""

from verl.single_controller.base.worker import DistRankInfo, DistGlobalInfo


def test_dist_global_info():
    dist_info = DistGlobalInfo(tp_size=2, cp_size=2, ep_size=2, dp_size=2, pp_size=2)

    result = []

    for i in range(dist_info.get_world_size()):
        result.append(dist_info.get_rank_info(global_rank=i, order='tp-pp-dp-ep-cp'))

    expected_result = [
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=1),
    ]

    for info, expected_info in zip(result, expected_result):
        assert info == expected_info
