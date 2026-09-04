from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.contracts.analysis_contract import AnalysisRequest
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    CodeAnalysisRequest,
    RepositoryAnalysisRequest,
    RepositorySecurityAnalysis,
    SandboxVerifyRequest,
    SecurityAnalysis,
    Vulnerability,
)

from app.services.featherless import FeatherlessService
from app.services.repository_service import RepositoryService
from app.services.security_agent import SecurityAgent
from app.analysis_engine.analyzers.analysis_engine import AnalysisEngine
from app.ai_engine.agents.ai_engine import AIEngine
from app.ai_engine.schemas.models import CodeInput
from app.sandbox.manager import SandboxManager

app = FastAPI(
    title="ShadowCode AI DevSecOps Platform",
    description="Unified AI-powered code security analysis, repository scanning, patch verification, and DevSecOps assistant",
    version="1.0.0",
)

# Enable CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ShadowCode AI DevSecOps Platform is running",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        service = FeatherlessService()
        system_prompt = (
            "You are ShadowCode, an expert DevSecOps AI assistant. Provide concise, "
            "accurate security advice, remediation suggestions, and code review insights."
        )
        response = await service.chat(request.message, system_prompt=system_prompt)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=SecurityAnalysis)
@app.post("/analyze/code", response_model=SecurityAnalysis)
async def analyze_code(request: CodeAnalysisRequest):
    try:
        # 1. Run Static Analysis Engine (tree-sitter)
        analysis_engine = AnalysisEngine()
        static_req = AnalysisRequest(
            language=request.language,
            file_path=request.file_path,
            code=request.code,
        )
        static_res = analysis_engine.analyze(static_req)

        # 2. Run Featherless AI Analysis
        vulnerabilities = []
        try:
            sec_agent = SecurityAgent()
            ai_vulnerability_res = await sec_agent.analyze_code(request.code)
            for v in ai_vulnerability_res.get("vulnerabilities", []):
                vulnerabilities.append(
                    Vulnerability(
                        name=v.get("name", "Security Vulnerability"),
                        severity=v.get("severity", "MEDIUM"),
                        description=v.get("description", ""),
                        evidence=v.get("evidence", ""),
                        impact=v.get("impact", ""),
                        remediation=v.get("remediation", ""),
                        confidence=v.get("confidence", "HIGH"),
                        line=v.get("line"),
                    )
                )
        except Exception:
            pass

        # Also add static findings as vulnerabilities if not already present
        for f in static_res.findings:
            vulnerabilities.append(
                Vulnerability(
                    name=f.title,
                    severity=f.severity,
                    description=f.description,
                    remediation=f.recommendation,
                    line=f.line,
                )
            )

        # 3. Combine with AI Decision Engine
        ai_engine = AIEngine()
        code_input = CodeInput(
            language=request.language,
            file_path=request.file_path,
            code=request.code,
        )
        final_data = ai_engine.analyze(code_input, static_res)

        return SecurityAnalysis(
            vulnerabilities=vulnerabilities,
            static_findings=[f.model_dump() for f in static_res.findings],
            risk_score=final_data["final_risk_score"],
            security_decision=final_data["security_decision"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/repository", response_model=RepositorySecurityAnalysis)
async def analyze_repository(request: RepositoryAnalysisRequest):
    repo_service = RepositoryService()
    security_agent = SecurityAgent()
    analysis_engine = AnalysisEngine()

    repository_path = None
    all_vulnerabilities = []
    severity_summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    files_analyzed = 0

    try:
        repository_path = repo_service.clone_repository(request.repository_url)
        source_files = repo_service.collect_source_files(repository_path)
        files_analyzed = len(source_files)

        for file_path in source_files:
            try:
                chunks = repo_service.read_file_chunks(file_path)
                for chunk in chunks:
                    # Static scan
                    static_req = AnalysisRequest(
                        language="python" if file_path.suffix == ".py" else "other",
                        file_path=file_path.name,
                        code=chunk,
                    )
                    static_res = analysis_engine.analyze(static_req)
                    for f in static_res.findings:
                        sev = f.severity.upper()
                        if sev in severity_summary:
                            severity_summary[sev] += 1
                        all_vulnerabilities.append(
                            Vulnerability(
                                name=f.title,
                                severity=f.severity,
                                description=f.description,
                                remediation=f.recommendation,
                                line=f.line,
                            )
                        )

                    # AI Scan
                    try:
                        ai_res = await security_agent.analyze_code(chunk)
                        for v in ai_res.get("vulnerabilities", []):
                            sev = v.get("severity", "MEDIUM").upper()
                            if sev in severity_summary:
                                severity_summary[sev] += 1
                            all_vulnerabilities.append(
                                Vulnerability(
                                    name=v.get("name", "Vulnerability"),
                                    severity=v.get("severity", "MEDIUM"),
                                    description=v.get("description", ""),
                                    evidence=v.get("evidence", ""),
                                    impact=v.get("impact", ""),
                                    remediation=v.get("remediation", ""),
                                    confidence=v.get("confidence", "HIGH"),
                                )
                            )
                    except Exception:
                        pass
            except Exception:
                continue

        return RepositorySecurityAnalysis(
            repository_url=request.repository_url,
            files_analyzed=files_analyzed,
            total_vulnerabilities=len(all_vulnerabilities),
            severity_summary=severity_summary,
            vulnerabilities=all_vulnerabilities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if repository_path:
            repo_service.cleanup(repository_path)


@app.post("/sandbox/verify")
async def verify_patch(request: SandboxVerifyRequest):
    try:
        manager = SandboxManager()
        result = manager.verify_patch(
            repository_path=request.repository_path,
            patch_text=request.patch_text,
            image=request.image,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
