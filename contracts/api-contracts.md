# API Contracts: Physical AI & Humanoid Robotics Textbook

## Overview

This document defines the API contracts for the Physical AI & Humanoid Robotics textbook system. Since this is a documentation site built with Docusaurus, the "API" refers to the content API and plugin interfaces that allow for extensibility and integration.

## Content API

### 1. Document Content API

#### Get Document Content
- **Endpoint**: `/api/docs/{module}/{week}/{page}`
- **Method**: GET
- **Description**: Retrieves the content of a specific document page
- **Parameters**:
  - `module` (string, required): Module identifier (e.g., "module-1", "module-2")
  - `week` (string, required): Week identifier (e.g., "week-3", "week-8")
  - `page` (string, required): Page identifier (e.g., "introduction", "architecture")

- **Response**:
  ```
  {
    "id": "string",
    "title": "string",
    "content": "markdown_content",
    "module": "string",
    "week": "string",
    "learningObjectives": ["string"],
    "prerequisites": ["string"],
    "relatedPages": ["string"],
    "lastModified": "ISO_date_string",
    "metadata": {
      "author": "string",
      "difficulty": "beginner|intermediate|advanced",
      "estimatedReadingTime": "integer_minutes"
    }
  }
  ```

- **Status Codes**:
  - 200: Success
  - 404: Document not found
  - 500: Server error

#### Search Documents
- **Endpoint**: `/api/search`
- **Method**: GET
- **Description**: Search across all documentation content
- **Parameters**:
  - `q` (string, required): Search query
  - `module` (string, optional): Filter by module
  - `week` (string, optional): Filter by week
  - `limit` (integer, optional): Number of results (default: 10)

- **Response**:
  ```
  {
    "query": "string",
    "results": [
      {
        "id": "string",
        "title": "string",
        "module": "string",
        "week": "string",
        "page": "string",
        "preview": "string",
        "score": "float",
        "url": "string"
      }
    ],
    "totalResults": "integer",
    "hasMore": "boolean"
  }
  ```

### 2. Navigation API

#### Get Navigation Structure
- **Endpoint**: `/api/navigation`
- **Method**: GET
- **Description**: Retrieve the complete navigation structure
- **Response**:
  ```
  {
    "course": {
      "title": "Physical AI & Humanoid Robotics",
      "description": "A Comprehensive 13-Week Course on Embodied Intelligence",
      "modules": [
        {
          "id": "module-1",
          "title": "The Robotic Nervous System (ROS 2)",
          "weeks": [
            {
              "id": "week-3",
              "title": "ROS 2 Architecture",
              "pages": [
                {
                  "id": "architecture",
                  "title": "Week 3: ROS 2 Architecture",
                  "path": "/module1/week3/architecture"
                }
              ]
            }
          ]
        }
      ]
    }
  }
  ```

### 3. Course Progress API

#### Get User Progress
- **Endpoint**: `/api/progress`
- **Method**: GET
- **Description**: Retrieve user progress through the course
- **Headers**:
  - `Authorization: Bearer {token}` (if authentication is implemented)

- **Response**:
  ```
  {
    "userId": "string",
    "modulesCompleted": ["string"],
    "currentModule": "string",
    "weeksCompleted": ["string"],
    "currentPage": "string",
    "overallProgress": "float_percentage",
    "lastAccessed": "ISO_date_string",
    "timeSpent": "integer_minutes"
  }
  ```

#### Update User Progress
- **Endpoint**: `/api/progress`
- **Method**: POST
- **Description**: Update user progress through the course
- **Headers**:
  - `Authorization: Bearer {token}`
  - `Content-Type: application/json`

- **Request Body**:
  ```
  {
    "pageId": "string",
    "moduleId": "string",
    "weekId": "string",
    "completed": "boolean",
    "timeSpent": "integer_minutes"
  }
  ```

- **Response**:
  ```
  {
    "success": "boolean",
    "message": "string",
    "updatedProgress": {
      "overallProgress": "float_percentage",
      "modulesCompleted": ["string"],
      "weeksCompleted": ["string"]
    }
  }
  ```

### 4. Quiz/Assessment API

#### Get Module Assessment
- **Endpoint**: `/api/assessment/{moduleId}`
- **Method**: GET
- **Description**: Retrieve assessment questions for a specific module
- **Parameters**:
  - `moduleId` (string, required): Module identifier

- **Response**:
  ```
  {
    "moduleId": "string",
    "title": "string",
    "description": "string",
    "questions": [
      {
        "id": "string",
        "type": "multiple-choice|true-false|short-answer|coding",
        "question": "string",
        "options": ["string"], // for multiple choice
        "correctAnswer": "string|integer", // varies by type
        "explanation": "string",
        "difficulty": "beginner|intermediate|advanced"
      }
    ],
    "totalQuestions": "integer",
    "estimatedTime": "integer_minutes"
  }
  ```

#### Submit Assessment Answers
- **Endpoint**: `/api/assessment/{moduleId}/submit`
- **Method**: POST
- **Description**: Submit answers for a module assessment
- **Headers**:
  - `Content-Type: application/json`

- **Request Body**:
  ```
  {
    "userId": "string",
    "answers": [
      {
        "questionId": "string",
        "answer": "string|integer|boolean"
      }
    ]
  }
  ```

- **Response**:
  ```
  {
    "success": "boolean",
    "score": "float_percentage",
    "totalQuestions": "integer",
    "correctAnswers": "integer",
    "feedback": [
      {
        "questionId": "string",
        "correct": "boolean",
        "explanation": "string"
      }
    ]
  }
  ```

### 5. Resource API

#### Get Course Resources
- **Endpoint**: `/api/resources`
- **Method**: GET
- **Description**: Retrieve all course-related resources
- **Parameters**:
  - `module` (string, optional): Filter by module
  - `type` (string, optional): Filter by resource type (video, article, code, diagram)

- **Response**:
  ```
  {
    "resources": [
      {
        "id": "string",
        "title": "string",
        "description": "string",
        "type": "video|article|paper|code|diagram|tutorial",
        "url": "string",
        "moduleId": "string",
        "weekId": "string",
        "tags": ["string"],
        "difficulty": "beginner|intermediate|advanced",
        "duration": "integer_minutes" // for videos
      }
    ],
    "totalResources": "integer",
    "filters": {
      "types": ["string"],
      "modules": ["string"],
      "difficulties": ["string"]
    }
  }
  ```

### 6. Exercise API

#### Get Week Exercises
- **Endpoint**: `/api/exercises/{moduleId}/{weekId}`
- **Method**: GET
- **Description**: Retrieve exercises for a specific week
- **Parameters**:
  - `moduleId` (string, required): Module identifier
  - `weekId` (string, required): Week identifier

- **Response**:
  ```
  {
    "moduleId": "string",
    "weekId": "string",
    "exercises": [
      {
        "id": "string",
        "title": "string",
        "description": "string",
        "difficulty": "easy|medium|hard",
        "instructions": "markdown_content",
        "starterCode": "string", // for coding exercises
        "expectedOutput": "string",
        "solution": "string", // hidden in response, used for validation
        "hints": ["string"],
        "resources": ["string"], // related resource IDs
        "estimatedTime": "integer_minutes"
      }
    ],
    "totalExercises": "integer"
  }
  ```

#### Submit Exercise Solution
- **Endpoint**: `/api/exercises/{exerciseId}/submit`
- **Method**: POST
- **Description**: Submit solution for an exercise
- **Headers**:
  - `Content-Type: application/json`

- **Request Body**:
  ```
  {
    "userId": "string",
    "solution": "string", // code or text solution
    "exerciseId": "string"
  }
  ```

- **Response**:
  ```
  {
    "success": "boolean",
    "valid": "boolean",
    "feedback": "string",
    "score": "float_percentage",
    "improvementHints": ["string"]
  }
  ```

## Plugin Interfaces

### 1. Content Plugin Interface
```typescript
interface ContentPlugin {
  /**
   * Process content before rendering
   */
  processContent(content: string, metadata: DocumentMetadata): ProcessedContent;

  /**
   * Transform content for search indexing
   */
  transformForSearch(content: string): SearchIndexEntry;

  /**
   * Validate content structure
   */
  validateContent(content: string, schema: any): ValidationResult;
}
```

### 2. Diagram Plugin Interface
```typescript
interface DiagramPlugin {
  /**
   * Render diagram markup to HTML/SVG
   */
  renderDiagram(diagramCode: string, type: DiagramType): string;

  /**
   * Validate diagram syntax
   */
  validateDiagram(diagramCode: string): ValidationResult;

  /**
   * Export diagram to different formats
   */
  exportDiagram(diagramCode: string, format: ExportFormat): Buffer;
}
```

### 3. Search Plugin Interface
```typescript
interface SearchPlugin {
  /**
   * Index document content
   */
  indexDocument(doc: IndexedDocument): Promise<void>;

  /**
   * Search across indexed documents
   */
  search(query: string, options: SearchOptions): Promise<SearchResults>;

  /**
   * Update search index
   */
  updateIndex(updates: IndexUpdate[]): Promise<void>;

  /**
   * Clear search index
   */
  clearIndex(): Promise<void>;
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "string",
    "timestamp": "ISO_date_string",
    "path": "string",
    "requestId": "string"
  }
}
```

### Common Error Codes
- `DOC_NOT_FOUND`: Requested document does not exist
- `INVALID_INPUT`: Malformed request or invalid parameters
- `AUTH_REQUIRED`: Authentication required for access
- `RATE_LIMITED`: Too many requests from client
- `SERVER_ERROR`: Internal server error
- `VALIDATION_FAILED`: Content validation failed
- `PERMISSION_DENIED`: Insufficient permissions

## Authentication & Authorization

### Public Endpoints (No Auth Required)
- `/api/docs/*` - Reading documentation
- `/api/search` - Search functionality
- `/api/navigation` - Navigation structure
- `/api/resources` - Course resources

### Protected Endpoints (Auth Required)
- `/api/progress/*` - Progress tracking
- `/api/assessment/*/submit` - Submitting assessments
- `/api/exercises/*/submit` - Submitting exercises

## Rate Limiting

### API Rate Limits
- **Anonymous users**: 100 requests per hour per IP
- **Authenticated users**: 1000 requests per hour per user
- **Administrative endpoints**: 10 requests per minute

## Versioning

### API Versioning
- Current version: `v1` (implicit in all endpoints)
- Future versions: `/api/v2/...`, `/api/v3/...`
- Backward compatibility maintained for 12 months after deprecation

## Security Considerations

### Input Validation
- All user inputs must be sanitized
- Content must be validated against schema
- File uploads restricted to safe formats

### Data Protection
- User progress data encrypted at rest
- API tokens secured with proper expiration
- Audit logs for sensitive operations

This API contract provides a comprehensive interface for extending and integrating with the Physical AI & Humanoid Robotics textbook system while maintaining security, performance, and usability standards.