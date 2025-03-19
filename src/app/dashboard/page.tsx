// src/app/dashboard/page.tsx
import { redirect } from 'next/navigation';

export default function Dashboard() {
  // Simply redirect to overview
  redirect('/dashboard/overview');
}