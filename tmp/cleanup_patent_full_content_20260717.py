from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


sys.path.insert(0, "/Users/zds/bsr/tmp")
from revise_patent_full_content_20260717 import (  # noqa: E402
    NS,
    para_text,
    set_mixed,
    set_text,
)
from convert_patent_formula_omml import build_transform  # noqa: E402


TARGET = Path(
    "/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"
)


def find_one(paras, prefix):
    matches = [p for p in paras if para_text(p).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one paragraph for {prefix!r}, found {len(matches)}")
    return matches[0]


def patch():
    transform = build_transform()
    tmp = TARGET.with_suffix(".cleanup.tmp.docx")
    with ZipFile(TARGET, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        paras = root.findall(".//w:body/w:p", namespaces=NS)

        set_text(
            find_one(paras, "10. 根据权利要求1所述的方法"),
            "10. 根据权利要求1所述的方法，其特征在于，所述重建运行时元数据包括重建专家目录、刷新调度拓扑、重建通信进程组以及更新失效逻辑计算进程到替换逻辑计算进程的映射关系，并将替换逻辑计算进程依次置于恢复中、已修复、屏障等待和健康状态。",
            transform,
        )

        set_mixed(
            find_one(paras, "9. 根据权利要求1所述的方法"),
            [
                "9. 根据权利要求1所述的方法，其特征在于，恢复专家状态时采用权重优先、优化器稍后的两阶段协议："
                "第一阶段恢复非专家状态和专家权重，使替换逻辑计算进程能够进入前向传播和反向传播；"
                "第二阶段在后台恢复专家优化器状态，在所述专家优化器状态未附着前通过更新屏障阻止受影响专家的优化器提交并保留对应梯度，"
                "在所述专家优化器状态附着后处理所保留的梯度并释放更新屏障。"
            ],
            transform,
        )

        set_text(
            find_one(paras, "[0004] 在大规模训练运行过程中"),
            "[0004] 在大规模训练运行过程中，逻辑计算进程失效、图形处理器故障、节点重启、通信超时等事件难以完全避免。现有主流恢复方式通常采用检查点重启，即从最近一次全局一致检查点恢复训练状态，并重放从检查点步到故障步之间的全部迭代。该方式虽然语义简单，但会丢弃检查点之后已经完成的图形处理器计算，并产生较大的状态加载和迭代重放开销。",
            transform,
        )

        set_mixed(
            find_one(paras, "[0005] 对于密集模型训练而言"),
            [
                "[0005] 对于密集模型训练而言，如果失效逻辑计算进程的全部模型状态在数据并行维度上存在健康副本，则可以通过从健康对等进程同步状态的方式减少检查点重启开销。"
                "然而，稀疏混合专家模型的训练状态具有异构性：非专家层和路由器等状态通常在非专家层数据并行维度上复制，而专家权重及其优化器状态则沿专家并行维度或专家张量并行维度进行分片。"
                "在专家数据并行度（EDP）满足",
                ("math", r"EDP=1"),
                "的布局下，失效逻辑计算进程拥有的专家分片在当前内存中不存在健康副本。",
            ],
            transform,
        )

        set_text(
            find_one(paras, "[0007] 此外，若仅将非专家状态恢复到当前步"),
            "[0007] 此外，若仅将非专家状态恢复到当前步而将专家状态恢复到检查点步，则会引入专家状态相对于非专家状态的陈旧性。该陈旧性并非必然导致训练失败，但如果在多个逻辑计算进程或多个时间窗口内累积，可能影响专家负载均衡和训练轨迹。现有技术缺乏一种能够在运行时量化该陈旧性、据此判断是否允许混合恢复、并对恢复过程进行结构化审计的技术方案。",
            transform,
        )

        set_text(
            find_one(paras, "[0018] 步骤七，重集成与审计。"),
            "[0018] 步骤七，重集成与审计。替换逻辑计算进程依次经过恢复中、已修复、屏障等待和健康状态，恢复控制器生成结构化日志，记录恢复路径、阈值输入、回退原因、时延分段和状态迁移，以支持离线审计和质量关联分析。",
            transform,
        )

        set_text(
            find_one(paras, "[0019] 本发明还提供一种实现上述方法"),
            "[0019] 本发明还提供一种实现上述方法的运行时混合恢复系统，包括故障检测与替换逻辑计算进程管理模块、安全点控制模块、状态分类模块、恢复策略判定模块、状态恢复模块、两阶段恢复模块以及重集成与结构化日志模块。",
            transform,
        )

        set_mixed(
            find_one(paras, "[0014] 步骤三，专家来源确定与陈旧暴露计算。"),
            [
                "[0014] 步骤三，专家来源确定与陈旧暴露计算。根据健康专家对等进程的可用性，确定当前事件中拟从检查点恢复并产生陈旧暴露的专家集合",
                ("math", r"E_{\mathrm{ckpt}}(t)"),
                "；从健康专家对等进程同步到当前安全点的专家不计入该集合。计算检查点间隔",
                ("math", r"\Delta=t-c"),
                "，维护暴露窗口",
                ("math", r"[t-W_{\mathrm{exp}},t)"),
                "内历史混合恢复事件的专家陈旧债务",
                ("math", r"S(t)=\sum_h |E_h|\Delta_h"),
                "，并根据",
                ("math", r"\Phi'(t)=\frac{S(t)+|E_{\mathrm{ckpt}}(t)|\Delta}{N_{\mathrm{expert}}W_{\mathrm{exp}}}"),
                "计算当前事件若采用混合恢复时的专家加权陈旧密度。暴露域为共享同一专家陈旧预算的专家集合，",
                ("math", r"N_{\mathrm{expert}}"),
                "为该暴露域内专家总数；暴露域可以配置为训练作业中的全部专家，或者按照流水线阶段或专家并行组划分，并在同一暴露窗口内保持不变。",
            ],
            transform,
        )

        set_mixed(
            find_one(paras, "[0037] 在另一个实施例中，若多个逻辑计算进程同时失效"),
            [
                "[0037] 在另一个实施例中，若多个逻辑计算进程同时失效，系统首先检查每个失效逻辑计算进程是否仍存在符合条件的健康非专家层数据并行对等进程；"
                "若某一失效逻辑计算进程的全部候选非专家层数据并行对等进程均不可用，则该失效事件回退到检查点重启路径。"
                "对于其余失效事件，系统按照预设顺序或并行批处理方式，针对失效逻辑计算进程i确定拟从检查点恢复的专家集合",
                ("math", r"E_{\mathrm{ckpt},i}(t)"),
                "和检查点间隔",
                ("math", r"\Delta_i"),
                "，并将对应的",
                ("math", r"|E_{\mathrm{ckpt},i}(t)|\Delta_i"),
                "贡献纳入窗口债务。若累计",
                ("math", r"\Phi'(t)"),
                "超过阈值，则后续失效事件或整个批次回退到检查点重启路径，以避免多逻辑计算进程突发故障导致专家陈旧暴露过大。",
            ],
            transform,
        )

        xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        with ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = xml
                zout.writestr(info, data)

    tmp.replace(TARGET)


if __name__ == "__main__":
    patch()
    print(TARGET)
