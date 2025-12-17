# Feature Specification: AI-Native Book RAG Chatbot Integration

**Feature Branch**: `1-rag-chatbot`
**Created**: 2025-12-17
**Status**: Draft
**Input**: User description: "# Specification: AI-Native Book RAG Chatbot Integration

## Overview
Build an integrated RAG chatbot for the \"AI-Native Software Development\" book website (https://ai-native.panaversity.org/). The interface must mirror the site's professional, dark-themed, documentation-first aesthetic.

## Tech Stack
- **Frontend:** React/TypeScript with Tailwind CSS (mirroring Panaversity styles).
- **Backend:** FastAPI (Python).
- **Vector Database:** Qdrant Cloud (Free Tier).
- **Embeddings:** Cohere (embed-english-v3.0).
- **Metadata Store:** Neon Serverless Postgres.
- **SDK:** OpenAI Agents/ChatKit.

## Configuration Details
- **Qdrant Endpoint**: https://f7896955-bf33-46c5-b4e8-454d2cb95879.europe-west3-0.gcp.cloud.qdrant.io
- **Qdrant API Key**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6jyatLdOi0JSm_K75ZZb4BBVi_7YJ6WZKrrn_-IqSKM
- **Cohere API Key**: LR9E3trjxOQxStV0sDLqxyYKIVJyRbohsZlT9aPQ
- **Neon Postgres Connection**: `neon:psql 'postgresql://neondb_owner:npg_5CXpMAVgFtO1@ep-dark-haze-abairq5s-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'`

## Functional Requirements
1. **Hybrid Interface:**
   - A persistent floating chat button (Electric Blue) at the bottom-right.
   - A slide-out chat drawer with glassmorphism effects.
2. **Context-Aware Retrieval:**
   - Chatbot must query Qdrant using Cohere embeddings to retrieve book sections.
   - **Selection-Based RAG:** If a user highlights text on the page, the chatbot must offer an \"Explain Selection\" button that uses the highlighted text as the primary context for the next query. The system should answer questions based strictly on the selected/highlighted text.
3. **Citations:** Every answer must provide a source link or chapter reference from the book.
4. **Chat History:** Conversation history must be stored in Neon Postgres database for session persistence.

## UI Design Specs (from ai-native.panaversity.org)
- **Background:** #0f172a (Deep Navy).
- **Typography:** Inter/SF Pro with high readability.
- **Component Style:** Bordered cards with low-opacity backgrounds (#1e293b).
- **User/Agent Bubbles:** Rounded corners, distinct colors for clarity.

## Implementation Plan
- **Phase 1:** Setup FastAPI endpoints for `/search` and `/chat`.
- **Phase 2:** Integrate Cohere to generate 1024-dim vectors.
- **Phase 3:** Configure Qdrant collection with `Distance.COSINE`.
- **Phase 4:** Build the React chat component with selection-tracking hooks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Chatbot Interface (Priority: P1)

As a reader of the AI-Native Software Development book, I want to quickly access a chatbot that can answer questions about the book content so that I can get instant clarification on complex topics without navigating away from my current page.

**Why this priority**: This is the foundational user journey that enables all other interactions with the chatbot. Without easy access, users won't engage with the feature.

**Independent Test**: Can be fully tested by verifying that the floating chat button appears on all pages and opens the chat drawer when clicked, delivering immediate access to the chat functionality.

**Acceptance Scenarios**:

1. **Given** I am viewing any page of the book website, **When** I see the floating chat button at the bottom-right corner, **Then** I can click it to open the chat drawer
2. **Given** I have opened the chat drawer, **When** I close it, **Then** the floating chat button remains visible for re-access

---

### User Story 2 - Query Book Content (Priority: P1)

As a reader, I want to ask questions about specific book content and receive accurate answers with proper citations so that I can understand complex concepts and know exactly where the information comes from.

**Why this priority**: This is the core value proposition of the RAG chatbot - providing accurate, cited answers from the book.

**Independent Test**: Can be fully tested by submitting various queries to the chatbot and verifying that responses are relevant to the book content and include proper citations.

**Acceptance Scenarios**:

1. **Given** I have opened the chat interface, **When** I submit a question about book content, **Then** I receive a response with relevant information and source citations
2. **Given** I submitted a question, **When** the system cannot find relevant content, **Then** I receive a helpful response indicating the limitation

---

### User Story 3 - Explain Selected Text (Priority: P2)

As a reader, I want to highlight text on the page and use the chatbot to explain that specific content so that I can get deeper insights into passages I find confusing.

**Why this priority**: This provides enhanced contextual assistance beyond general queries, improving the learning experience.

**Independent Test**: Can be fully tested by selecting text on the page and using the "Explain Selection" feature to receive context-aware explanations.

**Acceptance Scenarios**:

1. **Given** I have selected text on the book page, **When** I click the "Explain Selection" button, **Then** the chatbot provides an explanation based primarily on the selected text
2. **Given** I have selected text, **When** I submit it for explanation, **Then** the system treats this as the primary context for the response

---

### Edge Cases

- What happens when the vector database is temporarily unavailable?
- How does the system handle very long text selections for explanation?
- What occurs when the user submits multiple rapid queries?
- How does the system behave when there are no relevant matches in the book content?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a persistent floating chat button positioned at the bottom-right of all pages
- **FR-002**: System MUST display a slide-out chat drawer with glassmorphism effects when the chat button is activated
- **FR-003**: System MUST allow users to submit text queries about book content
- **FR-004**: System MUST retrieve relevant book sections using vector similarity search against stored content
- **FR-005**: System MUST generate embeddings for user queries and compare them with stored book content embeddings
- **FR-006**: System MUST display responses with clear citation links to original book chapters/sections
- **FR-007**: System MUST detect text selection on the page and offer an "Explain Selection" button
- **FR-008**: System MUST prioritize the selected text as primary context when the "Explain Selection" button is used
- **FR-009**: System MUST maintain conversation history within a single session
- **FR-010**: System MUST follow the specified UI design specifications (dark theme, color scheme, typography)

### Key Entities

- **Book Content**: Represents the structured content of the AI-Native Software Development book, including chapters, sections, paragraphs with associated metadata for retrieval
- **User Query**: Represents the text input from users seeking information from the book content
- **Retrieved Context**: Represents the relevant book sections retrieved by the RAG system based on semantic similarity to the user query
- **Chat Response**: Represents the AI-generated response that incorporates retrieved context and provides proper citations

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can access the chat interface within 1 click from any page on the book website
- **SC-002**: 90% of user queries receive relevant responses with proper citations within 5 seconds
- **SC-003**: Users engage with the "Explain Selection" feature in at least 30% of sessions where text is selected
- **SC-004**: 85% of users find the chatbot responses helpful for understanding book content
- **SC-005**: The system successfully retrieves relevant content for 95% of valid queries