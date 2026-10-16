"""Shared fixtures. Every test runs offline — no network, no API key, no LLM."""

from __future__ import annotations

import pytest

from bookforge.config import Settings, reset_settings_cache
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    ChartSpec,
    ConceptNote,
    DiagramSpec,
    Exercise,
    FigureAsset,
    FrameAsset,
    GlossaryItem,
    TableAsset,
    TableSpec,
    TranscriptSegment,
    VideoRecord,
)
from bookforge.workspace import Workspace


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        compile_latex=False,
        openai_api_key="test-key-not-used",
        # Pinned so a developer's .env (e.g. a reasoning model, or no gateway
        # URL) cannot change what the LM-construction tests observe.
        openai_api_base="http://127.0.0.1:9/v1",
        analyst_model="openai/test-model",
        writer_model="openai/test-model",
        critic_model="openai/test-model",
        caption_model="openai/test-model",
        reflection_model="openai/test-model",
        qa_max_iterations=3,
        load_optimized=False,
    )


@pytest.fixture
def workspace(settings) -> Workspace:
    return Workspace(settings.workspace_root_abs, "test-channel")


@pytest.fixture
def video_record() -> VideoRecord:
    return VideoRecord(
        video_id="abc123XYZ_-",
        title="Kubernetes Networking Deep Dive",
        url="https://www.youtube.com/watch?v=abc123XYZ_-",
        duration_sec=1800,
        upload_date="20250101",
        chapter_number=1,
    )


@pytest.fixture
def transcript_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start_sec=0, text="Welcome to this deep dive on Kubernetes networking."),
        TranscriptSegment(start_sec=12, text="A cluster runs a flat network where every pod gets its own IP address."),
        TranscriptSegment(start_sec=30, text="The default MTU is 1500 bytes but overlay networks reduce it to 1450."),
        TranscriptSegment(start_sec=48, text="We measured 940 megabits per second on the host network."),
        TranscriptSegment(start_sec=65, text="With the overlay that dropped to 780 megabits per second."),
        TranscriptSegment(start_sec=90, text="Service discovery goes through CoreDNS which caches for 30 seconds."),
        TranscriptSegment(start_sec=120, text="Ingress controllers terminate TLS and route by hostname."),
        TranscriptSegment(start_sec=150, text="The kube-proxy component programs iptables rules on every node."),
    ]


@pytest.fixture
def transcript_text(transcript_segments) -> str:
    return "\n".join(f"[{s.timestamp}] {s.text}" for s in transcript_segments)


@pytest.fixture
def analysis() -> ChapterAnalysis:
    """A grounded analysis matching the transcript fixture."""
    return ChapterAnalysis(
        chapter_title="Kubernetes Cluster Networking",
        summary=(
            "Kubernetes gives every pod a routable IP on a flat cluster network. "
            "Overlay encapsulation costs both MTU and throughput, and service "
            "discovery is mediated by CoreDNS."
        ),
        learning_objectives=[
            "The reader can explain the flat pod network model.",
            "The reader can quantify the throughput cost of overlay networking.",
            "The reader can describe how kube-proxy programs iptables.",
        ],
        concepts=[
            ConceptNote(
                heading="The Flat Pod Network",
                explanation=(
                    "Every pod in a Kubernetes cluster receives its own IP address on a "
                    "flat network, so pods address each other directly without port "
                    "mapping. This removes a whole class of collisions that container "
                    "port mapping otherwise creates and simplifies service discovery. "
                    "Because the address space is flat, a pod on one node reaches a pod "
                    "on another node using the same address it would use locally, and "
                    "no translation layer sits between them. The cost of that guarantee "
                    "is pushed down into the network plugin, which must make every pod "
                    "address routable across the whole cluster."
                ),
                key_points=["Every pod gets an IP", "No port mapping required"],
                evidence_timestamps=["00:00:12"],
            ),
            ConceptNote(
                heading="Overlay Networking Costs",
                explanation=(
                    "Overlay networks encapsulate traffic, which reduces the usable MTU "
                    "from 1500 bytes to 1450 and cuts measured throughput from 940 to "
                    "780 megabits per second on the same hardware. That overhead is the "
                    "price paid for network portability across heterogeneous substrates. "
                    "Each encapsulated packet carries an outer header, so the payload "
                    "available to the application shrinks and more packets are needed "
                    "for the same volume of data. Operators who control the underlying "
                    "network can often avoid the penalty by routing pod addresses "
                    "natively instead of tunnelling them."
                ),
                key_points=["MTU drops to 1450", "Throughput falls to 780"],
                evidence_timestamps=["00:00:30", "00:00:48", "00:01:05"],
            ),
        ],
        tables=[
            TableSpec(
                title="Throughput comparison",
                caption="Measured throughput with and without overlay encapsulation.",
                headers=["Network mode", "Throughput (Mbps)"],
                rows=[["Host network", "940"], ["Overlay", "780"]],
            )
        ],
        charts=[
            ChartSpec(
                title="Throughput by network mode",
                caption="Overlay encapsulation reduces measured throughput.",
                chart_type="bar",
                x_label="Mode",
                y_label="Mbps",
                categories=["Host", "Overlay"],
                values=[940.0, 780.0],
                evidence_timestamps=["00:00:48", "00:01:05"],
            )
        ],
        diagrams=[
            DiagramSpec(
                title="Request path",
                caption="How an external request reaches a pod.",
                diagram_type="flowchart",
                elements=["Ingress", "Service", "Pod"],
                relationships=["Ingress -> Service : routes by hostname", "Service -> Pod"],
            )
        ],
        glossary=[GlossaryItem(term="CoreDNS", definition="The cluster DNS server.")],
        exercises=[
            Exercise(
                question="Why does overlay networking reduce the usable MTU?",
                answer="Encapsulation headers consume part of each frame.",
            )
        ],
    )


@pytest.fixture
def assets_manifest() -> AssetsManifest:
    return AssetsManifest(
        chapter_slug="01_kubernetes-networking-deep-dive",
        chapter_dir="chapters/01_kubernetes-networking-deep-dive",
        figures=[
            FigureAsset(
                filename="chapters/01_kubernetes-networking-deep-dive/figures/chart_1.pdf",
                kind="chart",
                caption="Overlay encapsulation reduces measured throughput.",
                label="fig:ch01-chart1",
            ),
            FigureAsset(
                filename="chapters/01_kubernetes-networking-deep-dive/figures/frame_01_30s.jpg",
                kind="frame",
                caption="The slide introducing the flat pod network model.",
                label="fig:ch01-frame1",
                timestamp_sec=30.0,
            ),
        ],
        tables=[
            TableAsset(
                tex_file="chapters/01_kubernetes-networking-deep-dive/tables/table_1.tex",
                caption="Measured throughput with and without overlay encapsulation.",
                label="tab:ch01-1",
            )
        ],
    )


@pytest.fixture
def frame_assets() -> list[FrameAsset]:
    return [
        FrameAsset(file=f"/tmp/frame_{i:05d}.jpg", timestamp_sec=i * 5.0,
                   phash=f"{i:016x}", sharpness=10.0 + i, contrast=20.0,
                   quality=0.5 + (i % 3) * 0.1)
        for i in range(1, 21)
    ]


def good_chapter_tex(manifest: AssetsManifest, declare: bool = True) -> str:
    """A chapter body that satisfies the current contract.

    Assets are placed by marker (\\bfFigure / \\bfTable), never by path — the
    declarations are generated from the manifest by BookForge, exactly as the
    pipeline does before writing chapter.tex.
    """
    from bookforge.design.layout import declaration_block

    figures_block = "\n\n".join(
        f"Figure~\\ref{{{fig.label}}} shows the relevant material.\n"
        f"\\bfFigure{{{fig.label}}}"
        for fig in manifest.figures
    )
    tables_block = "\n\n".join(
        f"Table~\\ref{{{tab.label}}} lists the measurements.\n"
        f"\\bfTable{{{tab.label}}}"
        for tab in manifest.tables
    )
    filler = (
        "Cluster networking rests on a small number of guarantees that the "
        "platform makes to every workload it schedules, and understanding them "
        "is what separates confident operation from guesswork. "
    ) * 40
    body = f"""\\chapter{{Kubernetes Cluster Networking}}\\label{{ch:01-kubernetes}}

\\bfsection{{Overview}}
Kubernetes gives every pod a routable IP on a flat cluster network. {filler}

\\bfsection{{Learning Objectives}}
\\begin{{itemize}}
  \\item The reader can explain the flat pod network model.
  \\item The reader can quantify the throughput cost of overlay networking.
\\end{{itemize}}

\\section{{The Flat Pod Network}}
Every pod receives its own IP address. {filler}

\\begin{{keyidea}}
Pods are first-class network citizens; the plugin makes that true.
\\end{{keyidea}}

\\section{{Overlay Networking Costs}}
Encapsulation reduces the MTU to 1450 bytes and throughput from 940 to 780
megabits per second. {filler}

\\section{{Key Data and Comparisons}}
{tables_block}

{figures_block}

\\section{{Conceptual Models}}
\\bfDiagram{{How an external request reaches a pod.}}{{fig:ch01-diagram1}}{{%
\\begin{{tikzpicture}}[node distance=1.2cm]
\\node[modernbox] (ingress) {{Ingress}};
\\node[neutralbox, below=1.2cm of ingress] (service) {{Service}};
\\node[accentbox, below=1.2cm of service] (pod) {{Pod}};
\\draw[flowarrow] (ingress) -- (service);
\\draw[flowarrow] (service) -- (pod);
\\end{{tikzpicture}}}}

\\bfsection{{Key Takeaways}}
\\begin{{itemize}}
  \\item Pods are first-class network citizens.
\\end{{itemize}}

\\bfsection{{Glossary}}
\\begin{{description}}
  \\item[CoreDNS] The cluster DNS server.
  \\item[CNI] Container Network Interface plugin that wires pod networking.
  \\item[Overlay] Encapsulated network that carries pod traffic across hosts.
  \\item[MTU] Maximum transmission unit of a network frame.
  \\item[Service] Stable virtual IP that load-balances to pods.
  \\item[Ingress] HTTP routing entrypoint into the cluster.
\\end{{description}}

\\bfsection{{Exercises}}
\\begin{{enumerate}}
  \\item Why does overlay networking reduce the usable MTU?
  \\item What guarantee does the flat pod network give every workload?
  \\item When would you prefer a non-overlay CNI?
  \\item How does a Service reach a Pod across nodes?
\\end{{enumerate}}

\\bfsection{{Solutions}}
\\begin{{enumerate}}
  \\item Encapsulation headers consume part of each frame.
  \\item Every pod receives a routable IP on a flat cluster network.
  \\item When you need full line-rate throughput without encapsulation overhead.
  \\item The Service VIP is load-balanced to pod IPs via kube-proxy or eBPF.
\\end{{enumerate}}
"""
    if declare:
        return declaration_block(manifest) + "\n" + body
    return body
