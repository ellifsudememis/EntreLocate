import asyncio
import json
import logging
import pandas as pd
import nest_asyncio
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os
from dotenv import load_dotenv
import pymupdf
load_dotenv()

# —————————————————— logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
handler.setFormatter(fmt)
logger.addHandler(handler)

# —————————————————— token counter helper
def count_tokens(model_name: str, prompt: str):
    model = genai.GenerativeModel(model_name)
    return model.count_tokens(prompt=prompt)

# —————————————————— Agent base class
_logged_agent_names = set() ## module-level cache -> ja ja

class Agent:
    generation_config = {
        "temperature": 0.1, # Low randomness
        "top_p": 0.95, # Top 95% token mass
        "top_k": 64, # Pick from top 64 tokens
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    def __init__(self, name, role, model_name="gemini-1.5-flash"):
        self.name = name
        self.role = role
        self.model_name = model_name 
        self.model = genai.GenerativeModel(model_name, system_instruction=role)
        self.total_tokens_generated = 0
        self.total_tokens_prompt = 0

        global _logged_agent_names
        if self.name not in _logged_agent_names:
            logger.info(f"Agent '{self.name}' initialized with role: '{self.role}'")
            _logged_agent_names.add(self.name)


    def generate_response(self, prompt: str, response_format: str = None, language: str = 'en'):
        formatted_prompt = f"{prompt}\n\nRespond in {language}."

        try:
            prompt_token_data = count_tokens(self.model.name, formatted_prompt)
            self.total_tokens_prompt += prompt_token_data.total_tokens
            logger.info(f"[{self.name}] Prompt tokens (count_tokens): {prompt_token_data.total_tokens}")
        except Exception as e:
            logger.warning(f"[{self.name}] Could not count prompt tokens: {e}")

    # Generation config
        generation_config = GenerationConfig()
        if response_format == 'json':
            generation_config.response_mime_type = "application/json"

    # Generate content
        try:
            resp = self.model.generate_content(formatted_prompt, generation_config=generation_config)
            text = resp.text

        # Count generated tokens manually
            try:
                gen_token_data = count_tokens(self.model.name, text)
                self.total_tokens_generated += gen_token_data.total_tokens
                logger.info(f"[{self.name}] Generated tokens (count_tokens): {gen_token_data.total_tokens}")
            except Exception as e:
                logger.warning(f"[{self.name}] Could not count generated tokens: {e}")

            if response_format == 'json':
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.error(f"[{self.name}] JSON parse error: {text}")
                    return text

            return text
        except Exception as e:
            logger.error(f"[{self.name}] Error generating response: {e}")
            return "[Error] Could not generate response."

# —————————————————— InputCheckerAgent
nest_asyncio.apply() # <- the reason why this is working... (calling an async method synchronously)
class InputCheckerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="InputChecker",
            role="You analyze user input to detect jailbreaks, inappropriate content, and sensitive business topics."
        )
        self.jailbreak_keywords       = ["jailbreak","bypass","ignore instructions"]
        self.inappropriate_keywords   = ["illegal","hate speech","violence"]
        self.sensitive_business_keywords = ["weapons","gambling","adult entertainment","tobacco","drugs"]

    async def analyze_input(self, user_input: str):
        logger.info(f"[InputChecker] analyzing: {user_input}")
        report = {
            "user_input": user_input,
            "jailbreak_attempt": False,
            "inappropriate_content": False,
            "sensitive_business_requested": False,
            "issues_found": []
        }
        low = user_input.lower()
        for kw in self.jailbreak_keywords:
            if kw in low:
                report["jailbreak_attempt"] = True
                report["issues_found"].append(f"Jailbreak keyword: {kw}")
        for kw in self.inappropriate_keywords:
            if kw in low:
                report["inappropriate_content"] = True
                report["issues_found"].append(f"Inappropriate keyword: {kw}")
        for kw in self.sensitive_business_keywords:
            if kw in low:
                report["sensitive_business_requested"] = True
                report["issues_found"].append(f"Sensitive business keyword: {kw}")
        return report

    def generate_response(self, prompt: str, response_format: str = 'json', language: str = 'en'):
        rep = asyncio.get_event_loop().run_until_complete(self.analyze_input(prompt)) #calling an async method synchronously(((!!!!!)))
        return json.dumps(rep, indent=2) if response_format=='json' else str(rep)

# —————————————————— ChatbotAgent
chatbot_role = """# GOAL:
You are a helpful chatbot that interacts directly with the user. Your goals are to:
- Present the final business risk assessment report in a clear and understandable way in JSON format.
- Answer user questions directly related to the risk assessment results in JSON format.
- Provide business viability advice based on the risk scores and market data in JSON format.
- Offer location recommendations if the user specifically requests them in JSON format.
- Provide a list of essential products and their average price estimates if requested in JSON format.
- Respond in the user's specified language (English or Turkish) in JSON format.
- Crucially: Only answer questions directly related to the business risk assessment in JSON format.

# RESPONSE FORMAT:
You MUST respond with a natural language query.

# EXAMPLE 1 (Presenting Report):
# INPUT: (JSON report from ReportGenerator)
# OUTPUT:
json
{
  "business_risk_assessment": {
    "location": "Şişli, Istanbul",
    "business_type": "Fırın",
    "summary": "...",
    "demand_outlook": "...",
    "competition_level": "...",
    "overall_risk": "Medium",
    "key_recommendations": ["...", "..."]
  }
}


# EXAMPLE 2 (Answering a question):
# INPUT: "What is the competition like?"
# OUTPUT:
json
{
  "competition_level": "The analysis of OpenStreetMap data indicates a high number of existing fırıns in Şişli."
}
"""

class ChatbotAgent(Agent):
    def __init__(self):
        super().__init__(name="Chatbot", role=chatbot_role)

    async def generate_response(self, prompt: str, response_format: str = None, language: str = 'en'):
        logger.info(f"[Chatbot] received prompt: {prompt}")
        await asyncio.sleep(1.0)

        response = super().generate_response(prompt, response_format, language)

        print(f"[Chatbot] Prompt tokens: {self.total_tokens_prompt}, Generated tokens: {self.total_tokens_generated}")

        return response

# —————————————————— Manager

import asyncio
import logging

logger = logging.getLogger("manager_logger")
logging.basicConfig(level=logging.INFO)

class Manager:
    def __init__(self, user_input: str):
        self.user_input = user_input
        self.rag = RAGRetrieverAgent()
        self.chatbot = ChatbotAgent()
        self.checker = InputCheckerAgent()
        self.validator = ValidatorAgent()
        self.analyzer = GeneralAnalyzerAgent()
        self.scorer = FinalScorerAgent()

    async def run(self):
        print(f"\n--- User says: {self.user_input!r} ---")

        # Step 1: Input validation
        print("\n[InputCheckerAgent Report]:")
        print(self.checker.generate_response(self.user_input))

        # Step 2: Retrieve RAG data
        rag_data = await self.rag.get_data(self.user_input)
        if not rag_data["data"]:
            return "[Manager] Failed to retrieve RAG data."

        # Step 3: Validate RAG data
        validation = await self.validator.validate(self.user_input, rag_data["data"])
        print("[Validator] RAG confidence:", validation)

        # Step 4: Analyze
        analysis = await self.analyzer.analyze(self.user_input, rag_data)


        # Step 5: Score
        score_result = await self.scorer.score(analysis["data"])

        # Step 6: Prepare report
        final_report = {
            "analysis_summary": analysis["data"],
            "final_score": score_result["data"]["final_score"],
            "explanation": json.loads(score_result["data"]["reasoning_steps"])
        }

        # Step 7: Generate chatbot reply using structured report
        prompt = f"""Generate a business risk report for this input: '{self.user_input}'.\nSummarize the following data:\n{json.dumps(final_report, indent=2)}"""
        response = await self.chatbot.generate_response(prompt)
        print("\n[Chatbot Final Summary]:\n", response)
        return response




# —————————————————— ERRORHANDLING AND NETWORK ISSUES AGENT
class ErrorHandlingAndNetworkAgent(Agent):
    def __init__(self):
        super().__init__(name="ErrorHandler", role="Check for errors or missing agent data.")

    async def validate_data(self, responses):
        logger.info("[ErrorHandler] validating data...")
        print("[ErrorHandler] validate_data() called.")

        required_agents = {"RAGRetriever", "OpenStreetMap"}
        received_agents = set()
        issues = []

        for response in responses:
            name = response.get("name")
            data = response.get("data")
            print(f"[ErrorHandler] Checking response from agent: {name}")
            if name:
                received_agents.add(name)
                if not data or data == "":
                    issues.append(f"No data from {name}")
                    print(f"[ErrorHandler] Issue found: No data from {name}")
            else:
                issues.append("Response missing 'name' field")
                print("[ErrorHandler] Issue found: Response missing 'name' field")

        missing_agents = required_agents - received_agents
        if missing_agents:
            issues.append(f"Missing data from: {', '.join(missing_agents)}")
            print(f"[ErrorHandler] Issue found: Missing agents - {', '.join(missing_agents)}")

        if issues:
            print("[ErrorHandler] Validation failed with issues.")
            return {"status": "error", "issues": issues}
        print("[ErrorHandler] All required agent data present.")
        return {"status": "ok"}

# —————————————————— Open Street map agent - API
class OpenStreetMapAgent(Agent):
    def __init__(self):
        super().__init__(name="OpenStreetMap", role="Get location-based business data.")

    async def get_data(self):
        logger.info(f"[{self.name}] retrieving data...")
        print(f"[{self.name}] get_data() called.")
        await asyncio.sleep(1.0)


        return {
            "name": self.name,
            "data": {
                "businesses": {
                    "fırın": 15,
                    "bakkal": 5,
                    "kafe": 10
                },
                "district": "şişli",
                "business_type": "fırın"
            }
        }



# —————————————————— RAG Retriever
class RAGRetrieverAgent:
    def __init__(self):
        import os
        import pandas as pd
        import pymupdf
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb import PersistentClient
        from chromadb.utils import embedding_functions

        # Paths
        self.DB_PATH = './chromaDB/'
        self.POPULATION_FILE = 'static/content/Kurulan_Kapanan_İş_Yeri_İstatistikleri_Şubat_2025.pdf'
        self.EXPENDITURE_FILE = 'static/content/hanehalki tuketim harcamasinin turlerine gore dagilimi.xls'
        self.SEGREGATION_FILE = 'static/content/2022-ilce-sege.pdf'
        self.DEMOGRAPHICS_FILE_ORIG = 'static/content/yaş cinsiyete göre illere göre nüfus.xls'
        self.DEMOGRAPHICS_FILE = 'static/content/demographics.xls'

        if os.path.exists(self.DEMOGRAPHICS_FILE_ORIG):
            os.rename(self.DEMOGRAPHICS_FILE_ORIG, self.DEMOGRAPHICS_FILE)
            print(f"Renamed: {self.DEMOGRAPHICS_FILE_ORIG} → {self.DEMOGRAPHICS_FILE}")

        # Global buffers
        self.all_texts = []
        self.all_metadatas = []

        # Process files
        self._process_pdf(self.POPULATION_FILE, 'population')
        self._process_pdf(self.SEGREGATION_FILE, 'segregation')
        self._process_excel(self.EXPENDITURE_FILE, 'expenditure')
        self._process_excel(self.DEMOGRAPHICS_FILE, 'demographics')

        print(f"Finished reading files. {len(self.all_texts)} chunks created.")

        # Embedding and Collection
        self.COLLECTION_NAME = "EntreLocate"
        self.client = PersistentClient(path=self.DB_PATH)
        self.collection = self.client.get_or_create_collection(
            self.COLLECTION_NAME,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )

        if self.all_texts:
            print("Starting embedding...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(self.all_texts, batch_size=32, show_progress_bar=True)
            ids = [str(i) for i in range(len(self.all_texts))]
            try:
                existing_ids = self.collection.get()['ids']
                print(f"Deleting existing {len(existing_ids)} entries from collection...")
                self.collection.delete(ids=self.collection.get()['ids'])
            except:
                print("No existing entries found or failed to delete.")
                #pass
            for i in range(0, len(ids), 1000):
                self.collection.add(
                    documents=self.all_texts[i:i+1000],
                    embeddings=embeddings[i:i+1000],
                    metadatas=self.all_metadatas[i:i+1000],
                    ids=ids[i:i+1000]
                )
            print(f" Added {len(self.all_texts)} chunks to collection.")
        else:
            print(" No data to embed.")

    def _chunk_row_with_context(self, row_data, metadata, chunk_size=512):
        import pandas as pd
        row_text = ' '.join([str(cell) for cell in row_data if pd.notna(cell)])
        if not row_text.strip():
            return [], []
        sheet_info = f", Sheet: {metadata.get('sheet', '')}" if 'sheet' in metadata else ""
        context_text = f"Data from {metadata['source']}{sheet_info}, Category: {metadata['category']}. Row data: "
        full_text = context_text + row_text
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return chunks, [metadata] * len(chunks)

    def _process_pdf(self, file_path, category):
        try:
            print(f"Processing PDF: {file_path}")
            doc = pymupdf.open(file_path)
            print(f"  • Pages in {category}: {doc.page_count}")
            text = "".join([page.get_text() for page in doc])
            meta = {"source": file_path, "category": category}
            chunks, metas = self._chunk_row_with_context([text], meta)
            self.all_texts.extend(chunks)
            self.all_metadatas.extend(metas)

            for i, chunk in enumerate(chunks[:5]):
                print(f"    • Chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")

            print(f"{category.title()} data processed successfully.")
        except Exception as e:
            print(f"Error processing PDF ({category}): {e}")

    def _process_excel(self, file_path, category):
        try:
            print(f"Processing Excel: {file_path}")
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                for idx, row in df.iterrows():
                    meta = {"source": file_path, "sheet": sheet, "category": category, "row_index": idx}
                    chunks, metas = self._chunk_row_with_context(row.tolist(), meta)
                    self.all_texts.extend(chunks)
                    self.all_metadatas.extend(metas)

                    if idx < 5:
                        print(f"      • Chunked {category} row {idx+1} (sheet: {sheet}) → {len(chunks)} chunks")

            print(f"{category.title()} data processed successfully.")
        except Exception as e:
            print(f"Error processing Excel ({category}): {e}")

    def query(self, query_text, n_results=5):
        print(f"[RAGRetriever] Querying collection for: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results


    async def get_data(self, query_text="population and demographics"):
        print(f"[RAGRetriever] get_data() called with query: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
           n_results=10,
            include=["documents", "metadatas", "distances"]
        )

        print(f"[RAGRetriever] Retrieved {len(results['documents'][0])} results.")
        return {
            "name": "RAGRetriever",
           "data": results
       }




# —————————————————— General Analyzer
import json
import logging

logger = logging.getLogger(__name__)

# —————————————————— General Analyzer
class GeneralAnalyzerAgent(Agent):
    def __init__(self):
        super().__init__(name="GeneralAnalyzer", role="Analyze validated data.")

    async def analyze(self, user_input, rag_data):
        logger.info(f"[GeneralAnalyzer] Data received for analysis.")
        print("\n[GeneralAnalyzerAgent]: Data received and ready for analysis.")

        if not isinstance(rag_data, dict):
            raise TypeError(f"Expected rag_data to be dict, but got {type(rag_data).__name__}")

        try:
            socio_rank = rag_data.get("socio_economic_growth", {}).get("sira", None)
            opened = rag_data.get("newly_opened_closed_business", {}).get("opned", 0)
            closed = rag_data.get("newly_opened_closed_business", {}).get("closed", 0)
            closure_ratio = round(opened / closed, 2) if closed else None

            female_ratio = rag_data.get("population", {}).get("female population proportion", None)
            age_group = rag_data.get("population", {}).get("age group", None)

            result = {
                "socioeconomic_growth_rank": socio_rank,
                "socioeconomic_growth_analysis": (
                    f"Socioeconomic rank of {socio_rank} suggests a relatively "
                    f"{'strong' if socio_rank and socio_rank <= 200 else 'moderate' if socio_rank else 'low'} local economy."
                ) if socio_rank is not None else "No socioeconomic ranking available.",

                "new_business_closure_ratio": closure_ratio,
                "new_business_closure_analysis": (
                    f"New-to-closed business ratio of {closure_ratio}, suggesting "
                    f"{'a healthy environment' if closure_ratio and closure_ratio > 2 else 'a risky trend' if closure_ratio and closure_ratio < 1 else 'mixed outcomes'}."
                ) if closure_ratio is not None else "Business dynamics not analyzable due to missing data.",

                "relevant_population_ratio": female_ratio,
                "relevant_population_analysis": (
                    f"A female population ratio of {female_ratio} suggests a balanced demographic."
                ) if female_ratio is not None else "No population gender data available.",

                "relevant_age_group": age_group,
                "relevant_age_group_analysis": (
                    f"Dominant age group is {age_group}, indicating consumer profile."
                ) if age_group is not None else "Age group data missing for analysis.",

                "business_viability_assessment": "Overall, the district shows "
                    + ("strong potential" if closure_ratio and closure_ratio > 2 and socio_rank and socio_rank <= 200 else
                        "moderate potential" if closure_ratio and closure_ratio > 1 else
                        "high risk due to poor business dynamics or competition.")
            }

            return {
                "from": "GeneralAnalyzerAgent",
                "data": result
            }

        except Exception as e:
            logger.error(f"[GeneralAnalyzer] Error during analysis: {e}")
            return {
                "from": "GeneralAnalyzerAgent",
                "error": str(e)
            }

# —————————————————— FinalScorerAgent
import json
import logging

logger = logging.getLogger(__name__)

class FinalScorerAgent(Agent):
    def __init__(self):
        super().__init__(name="FinalScorer", role="Compute final risk score based on analysis data.")

    async def score(self, analysis_data):
        logger.info("[FinalScorer] Scoring started.")

        try:
            business_type = analysis_data.get("business_type", "general business")
            num_similar = analysis_data.get("num_similar_businesses")
            socio_rank = analysis_data.get("socioeconomic_growth_rank")
            closure_ratio = analysis_data.get("new_business_closure_ratio")
            pop_ratio = analysis_data.get("relevant_population_ratio")
            age_group = analysis_data.get("relevant_age_group")

            # --- Scoring Components ---
            # 1. Socioeconomic Growth Rank (0-20)
            if socio_rank is not None:
                socio_score = 20 * (1 - (socio_rank - 1) / 972)
            else:
                socio_score = 0

            # 2. Relevant Population Ratio (0-15)
            # Simple heuristic: assume 1.0 is max relevance
            # You can refine based on business_type later
            if pop_ratio is not None:
                if business_type.lower() in ["bakery", "fırın", "salon", "clothing store", "healthcare"]:
                    relevance_score = pop_ratio  # assuming women more relevant
                else:
                    relevance_score = 1  # fallback
                pop_score = 15 * min(1.0, relevance_score)
            else:
                pop_score = 0

            # 3. Business Dynamics (Open/Close Ratio) (0-15)
            if closure_ratio is not None:
                if closure_ratio >= 3:
                    dynamics_score = 15
                elif closure_ratio >= 1.5:
                    dynamics_score = 12
                elif closure_ratio >= 1.0:
                    dynamics_score = 9
                else:
                    dynamics_score = 5
            else:
                dynamics_score = 0

            # 4. Competition (Number of Similar Businesses) (0–50)
            if num_similar is not None:
                if num_similar == 0:
                    rivalry_score = 50
                    normalized_competition_score = 0
                elif num_similar <= 3:
                    rivalry_score = 45
                    normalized_competition_score = 0.1
                elif num_similar <= 10:
                    rivalry_score = 35
                    normalized_competition_score = 0.3
                elif num_similar <= 20:
                    rivalry_score = 25
                    normalized_competition_score = 0.5
                else:
                    rivalry_score = 10
                    normalized_competition_score = 0.8
            else:
                rivalry_score = 20
                normalized_competition_score = 0.6

            # Final score
            total_score = round(socio_score + pop_score + dynamics_score + rivalry_score)
            reasoning = {
                "demand_score": {
                    "socioeconomic_growth": f"Rank {socio_rank} contributes {round(socio_score, 2)} points.",
                    "relevant_demographics": f"Population ratio {pop_ratio} contributes {round(pop_score, 2)} points.",
                    "sector_dynamics": f"New-to-closed ratio {closure_ratio} contributes {round(dynamics_score, 2)} points."
                },
                "rivalry_score": {
                    "similar_businesses": f"{num_similar} similar businesses, normalized competition score = {normalized_competition_score}, contributes {round(rivalry_score, 2)} points."
                },
                "final_calculation": f"Total score = {round(socio_score, 2)} + {round(pop_score, 2)} + {round(dynamics_score, 2)} + {round(rivalry_score, 2)} = {total_score}"
            }

            return {
                "from": "FinalScorerAgent",
                "data": {
                    "final_score": total_score,
                    "reasoning_steps": json.dumps(reasoning, indent=2)
                }
            }

        except Exception as e:
            logger.error(f"[FinalScorer] Error during scoring: {e}")
            return {
                "from": "FinalScorerAgent",
                "error": str(e)
            }
            
def format(report: dict) -> str:
    try:
        assessment = report["business_risk_assessment"]
        html = f"""
        <h2>Business Risk Assessment Report</h2>
        <p><strong>Location:</strong> {assessment['location']}</p>
        <p><strong>Business Type:</strong> {assessment['business_type']}</p>
        <p><strong>Summary:</strong> {assessment['summary']}</p>
        <p><strong>Demand Outlook:</strong> {assessment['demand_outlook']}</p>
        <p><strong>Competition Level:</strong> {assessment['competition_level']}</p>
        <p><strong>Overall Risk:</strong> <span style="color:red;">{assessment['overall_risk']}</span></p>
        <h4>Key Recommendations:</h4>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in assessment['key_recommendations'])}
        </ul>
        """
        return html
    except Exception as e:
        logger.error(f"Failed to format report: {e}")
        return "<p>Error formatting report.</p>"
            


# —————————————————— SERP API AGENT

# —————————————————— Other SERP agent -> web search için

# —————————————————— REPORT GENERATOR AGENT

# —————————————————— VİSUALİZER AGENT

# —————————————————— VALİDATOR AGENT
from sentence_transformers import SentenceTransformer, util

class ValidatorAgent(Agent):
    def __init__(self):
        super().__init__(name="Validator", role="Validate if agent data matches user intent.")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def compute_confidence(self, user_input, data):
        if isinstance(data, dict) and "documents" in data:
            documents = data["documents"]
            if not documents or not any(documents[0]):
                return 0.0

            user_embedding = self.model.encode(user_input, convert_to_tensor=True)
            doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
            cosine_scores = util.cos_sim(user_embedding, doc_embeddings)[0]
            return round(float(cosine_scores.max()), 2)
        return 0.0

    async def validate(self, user_input, rag_data):
        logger.info("[Validator] Validating relevance of data to user input...")
        print("[Validator] validate() called. Calculating confidence scores...")

        rag_confidence = self.compute_confidence(user_input, rag_data)
        logger.info(f"[Validator] RAG confidence: {rag_confidence}")
        print(f"[Validator] RAG confidence: {rag_confidence}")

        return {
            "status": "ok" if rag_confidence > 0.5 else "not_good",
            "rag_confidence": rag_confidence,
            "user_input": user_input,
            "rag_data": rag_data
        }
        
        


# —————————————————— Run
if __name__ == "__main__":
    import os

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)