import typing
import logging
from enum import Enum, auto

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from .dataset import Dataset, DatasetManager


log = logging.getLogger(__name__)

Case = typing.TypeVar("Case")


class CaseType(Enum):
    """
    Example:
        >>> case_cls = CaseType.CapacityDim128.case_cls
        >>> assert c is not None
        >>> CaseType.CapacityDim128.case_name
        "Capacity Test (128 Dim Repeated)"
    """

    CapacityDim128 = 1
    CapacityDim960 = 2

    Performance768D100M = 3
    Performance768D10M = 4
    Performance768D1M = 5

    Performance768D10M1P = 6
    Performance768D1M1P = 7
    Performance768D10M99P = 8
    Performance768D1M99P = 9

    Performance1536D500K = 10
    Performance1536D5M = 11

    Performance1536D500K1P = 12
    Performance1536D5M1P = 13
    Performance1536D500K99P = 14
    Performance1536D5M99P = 15
    
    Performance960D100K90P=16
    Performance128D500K90P=17
    
    Performance960D100K80P=18
    Performance128D500K80P=19
    
    Performance960D100K70P=20
    Performance128D500K70P=21
    
    Performance960D100K60P=22
    Performance128D500K60P=23
    
    Performance960D100K50P=24
    Performance128D500K50P=25
    
    Performance960D100K40P=26
    Performance128D500K40P=27
    
    Performance960D100K30P=28
    Performance128D500K30P=29
    
    Performance960D100K20P=30
    Performance128D500K20P=31
    
    Performance960D100K10P=32
    Performance128D500K10P=33

    Custom = 100

    @property
    def case_cls(self, custom_configs: dict | None = None) -> Case:
        return type2case.get(self)

    @property
    def case_name(self) -> str:
        c = self.case_cls
        if c is not None:
            return c().name
        raise ValueError("Case unsupported")

    @property
    def case_description(self) -> str:
        c = self.case_cls
        if c is not None:
            return c().description
        raise ValueError("Case unsupported")


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """Undifined case

    Fields:
        case_id(CaseType): default 9 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | None): one of 99% | 1% | None
        filters(dict | None): filters for search
    """

    case_id: CaseType
    label: CaseLabel
    name: str
    description: str
    dataset: DatasetManager

    load_timeout: float | int
    optimize_timeout: float | int | None

    filter_rate: float | None

    @property
    def filters(self) -> dict | None:
        if self.filter_rate is not None:
            ID = round(self.filter_rate * self.dataset.data.size)
            return {
                "metadata": f">={ID}",
                "id": ID,
            }

        return None


class CapacityCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Load
    filter_rate: float | None = None
    load_timeout: float | int = config.CAPACITY_TIMEOUT_IN_SECONDS
    optimize_timeout: float | int | None = None


class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | None = None
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class CapacityDim960(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim960
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class CapacityDim128(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim128
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class Performance768D10M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D100M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.LAION.manager(100_000_000)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_100M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_100M


class Performance1536D500K(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Search Performance Test (500K Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 500K dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Search Performance Test (5M Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 5M dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D500K1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K

class Performance1536D5M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M

class Performance1536D500K99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K

class Performance1536D5M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M

class Performance960D100K90P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K90P
    filter_rate: float | int | None = 0.90
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 90%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>90% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K90P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K90P
    filter_rate: float | int | None = 0.90
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 90%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>90% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K80P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K80P
    filter_rate: float | int | None = 0.80
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 80%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>80% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K80P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K80P
    filter_rate: float | int | None = 0.80
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 80%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>80% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K70P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K70P
    filter_rate: float | int | None = 0.70
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 70%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>70% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K70P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K70P
    filter_rate: float | int | None = 0.70
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 70%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>70% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K60P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K60P
    filter_rate: float | int | None = 0.60
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 60%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>60% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K60P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K60P
    filter_rate: float | int | None = 0.60
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 60%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>60% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K50P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K50P
    filter_rate: float | int | None = 0.50
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 50%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>50% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K50P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K50P
    filter_rate: float | int | None = 0.50
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 50%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>50% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K40P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K40P
    filter_rate: float | int | None = 0.40
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 40%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>40% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K40P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K40P
    filter_rate: float | int | None = 0.40
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 40%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>40% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance960D100K30P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K30P
    filter_rate: float | int | None = 0.30
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 30%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>30% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K30P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K30P
    filter_rate: float | int | None = 0.30
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 30%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>30% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
class Performance960D100K20P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K20P
    filter_rate: float | int | None = 0.20
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 20%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>20% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K20P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K20P
    filter_rate: float | int | None = 0.20
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 20%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>20% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance960D100K10P(PerformanceCase):
    case_id: CaseType = CaseType.Performance960D100K10P
    filter_rate: float | int | None = 0.10
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 960 Dim, Filter 10%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Gist 100k vectors</b>, 960 dimensions) under a filtering rate (<b>10% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT

class Performance128D500K10P(PerformanceCase):
    case_id: CaseType = CaseType.Performance128D500K10P
    filter_rate: float | int | None = 0.10
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 128 Dim, Filter 10%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Sift 500k vectors</b>, 128 dimensions) under a filtering rate (<b>10% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT
    
type2case = {
    CaseType.CapacityDim960: CapacityDim960,
    CaseType.CapacityDim128: CapacityDim128,

    CaseType.Performance768D100M: Performance768D100M,
    CaseType.Performance768D10M: Performance768D10M,
    CaseType.Performance768D1M: Performance768D1M,

    CaseType.Performance768D10M1P: Performance768D10M1P,
    CaseType.Performance768D1M1P: Performance768D1M1P,
    CaseType.Performance768D10M99P: Performance768D10M99P,
    CaseType.Performance768D1M99P: Performance768D1M99P,

    CaseType.Performance1536D500K: Performance1536D500K,
    CaseType.Performance1536D5M: Performance1536D5M,

    CaseType.Performance1536D500K1P: Performance1536D500K1P,
    CaseType.Performance1536D5M1P: Performance1536D5M1P,

    CaseType.Performance1536D500K99P: Performance1536D500K99P,
    CaseType.Performance1536D5M99P: Performance1536D5M99P,
    
    CaseType.Performance960D100K90P: Performance960D100K90P,
    CaseType.Performance128D500K90P: Performance128D500K90P,
    
    CaseType.Performance960D100K80P: Performance960D100K80P,
    CaseType.Performance128D500K80P: Performance128D500K80P,
    
    CaseType.Performance960D100K70P: Performance960D100K70P,
    CaseType.Performance128D500K70P: Performance128D500K70P,
    
    CaseType.Performance960D100K60P: Performance960D100K60P,
    CaseType.Performance128D500K60P: Performance128D500K60P,
    
    CaseType.Performance960D100K50P: Performance960D100K50P,
    CaseType.Performance128D500K50P: Performance128D500K50P,
    
    CaseType.Performance960D100K40P: Performance960D100K40P,
    CaseType.Performance128D500K40P: Performance128D500K40P,
    
    CaseType.Performance960D100K30P: Performance960D100K30P,
    CaseType.Performance128D500K30P: Performance128D500K30P,
    
    CaseType.Performance960D100K20P: Performance960D100K20P,
    CaseType.Performance128D500K20P: Performance128D500K20P,
    
    CaseType.Performance960D100K10P: Performance960D100K10P,
    CaseType.Performance128D500K10P: Performance128D500K10P

}
