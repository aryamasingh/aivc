SAMPLE_NOTES = """Met founder of MediFlow AI after intro from a hospital CFO.

Company is building an AI workflow platform for prior authorization and utilization management.
Founder was formerly a product executive at Epic and says the team has deep revenue cycle experience.
Early users appear to be regional hospital systems and specialty groups.
Interesting because healthcare administrative cost is enormous and prior authorization remains painful for providers.
Claims early pilots reduced manual review time materially, but numbers are still self-reported.
Need to understand buyer urgency, integration burden, competitive intensity, and whether payers or providers capture the value.
"""


MOCK_EVIDENCE = [
    {
        "title": "MediFlow AI company website",
        "url": "https://example.com/mediflow",
        "snippet": "MediFlow AI describes a prior authorization automation platform for health systems and specialty practices.",
        "query": "MediFlow AI company prior authorization",
        "evidence_type": "website",
        "confidence": 0.72,
    },
    {
        "title": "Prior authorization automation market overview",
        "url": "https://example.com/prior-auth-market",
        "snippet": "Healthcare organizations are investing in automation to reduce administrative burden around prior authorization workflows.",
        "query": "prior authorization automation market size healthcare",
        "evidence_type": "market",
        "confidence": 0.68,
    },
    {
        "title": "Cohere Health raises growth funding",
        "url": "https://example.com/cohere-health-funding",
        "snippet": "Cohere Health is cited as a prior authorization and utilization management automation company serving health plans.",
        "query": "prior authorization automation competitors Cohere Health",
        "evidence_type": "competitor",
        "confidence": 0.7,
    },
    {
        "title": "AKASA healthcare automation profile",
        "url": "https://example.com/akasa-profile",
        "snippet": "AKASA offers AI-enabled healthcare operations automation for revenue cycle and administrative workflows.",
        "query": "healthcare administrative automation competitors AKASA",
        "evidence_type": "competitor",
        "confidence": 0.65,
    },
    {
        "title": "Healthcare AI adoption news",
        "url": "https://example.com/healthcare-ai-news",
        "snippet": "Hospital systems continue to test AI tools that reduce administrative workload while demanding evidence of ROI and integration feasibility.",
        "query": "hospital systems AI administrative automation news",
        "evidence_type": "news",
        "confidence": 0.62,
    },
    {
        "title": "Harvey AI company website",
        "url": "https://example.com/harvey-ai",
        "snippet": "Harvey is described as an AI platform for legal professionals and law firms, supporting legal research, drafting, and workflow automation.",
        "query": "Harvey AI legal professionals law firms",
        "evidence_type": "website",
        "confidence": 0.78,
    },
    {
        "title": "Legal AI workflow market overview",
        "url": "https://example.com/legal-ai-market",
        "snippet": "Law firms and legal departments are adopting AI tools for research, drafting, document review, and workflow automation.",
        "query": "AI legal workflow platform market law firms",
        "evidence_type": "market",
        "confidence": 0.7,
    },
    {
        "title": "Legal AI platform competitors",
        "url": "https://example.com/legal-ai-competitors",
        "snippet": "AI legal workflow vendors include Thomson Reuters CoCounsel, Legora, Spellbook, Eve, Lexion, Ironclad, and other legal technology platforms.",
        "query": "AI legal workflow platform competitors CoCounsel Legora Spellbook",
        "evidence_type": "competitor",
        "confidence": 0.72,
    },
    {
        "title": "Harvey law firm adoption news",
        "url": "https://example.com/harvey-law-firm-adoption",
        "snippet": "Large law firms are testing and deploying legal AI platforms while scrutinizing accuracy, confidentiality, and defensibility.",
        "query": "Harvey law firm adoption legal AI news",
        "evidence_type": "news",
        "confidence": 0.66,
    },
    {
        "title": "Collaborative design platform company website",
        "url": "https://example.com/collaborative-design-platform",
        "snippet": "The company describes a collaborative design platform used for interface design, prototyping, design systems, developer handoff, and product collaboration.",
        "query": "collaborative design platform product design company website",
        "evidence_type": "website",
        "confidence": 0.82,
    },
    {
        "title": "Collaborative product design software market overview",
        "url": "https://example.com/product-design-software-market",
        "snippet": "Product design teams increasingly use cloud collaboration, design systems, prototyping, and developer handoff tools to coordinate software development.",
        "query": "collaborative design software product design platform market",
        "evidence_type": "market",
        "confidence": 0.7,
    },
    {
        "title": "Collaborative design software competitive landscape",
        "url": "https://example.com/collaborative-design-competitors",
        "snippet": "Collaborative product design platforms can compete with Adobe, Canva, Miro, Sketch, Framer, and broader product collaboration or design-system tooling.",
        "query": "collaborative design software competitors Adobe Canva Miro Sketch Framer Atlassian",
        "evidence_type": "competitor",
        "confidence": 0.74,
    },
    {
        "title": "Collaborative design enterprise expansion and AI roadmap news",
        "url": "https://example.com/collaborative-design-enterprise-ai",
        "snippet": "Collaborative design platforms continue to expand enterprise collaboration features while product teams evaluate AI-assisted design, prototyping, and workflow automation.",
        "query": "collaborative design enterprise expansion AI roadmap news",
        "evidence_type": "news",
        "confidence": 0.68,
    },
    {
        "title": "Collaborative design pricing and enterprise plans",
        "url": "https://example.com/collaborative-design-pricing",
        "snippet": "Collaborative design software often uses team and enterprise plans with collaboration, administration, design systems, and security capabilities.",
        "query": "collaborative design pricing enterprise plans business model",
        "evidence_type": "business_model",
        "confidence": 0.66,
    },
]
